import torch
import itertools
import math
import tqdm
import psutil
import pandas as pd
import logging
import time

NUM_MAX_COMBINATION_PER_BOAT = int(1e5)
NUM_MAX_PAIRWISE_COMBINATION = int(1e5)

torch.set_printoptions(edgeitems=10)


#############################################
# Utility functions
#############################################
def unravel_indices(indices, shape):
    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def sample(x, k, replacement=False):
    """
    Sample k samples from x with or without replacement.

    if k > len(x), then we repeat the indices to fill the k samples
    (but guarantee that all indices are sampled at least once)
    """

    if replacement:
        return x[torch.randint(len(x), (k,))]
    else:
        if k > len(x):
            indices = torch.randperm(len(x))
            indices = indices.repeat(k // len(x) + 1)[: k - len(x)]
            indices = torch.cat([torch.arange(len(x)), indices])
            return x[indices]
        return x[torch.randperm(len(x))[:k]]


def generalized_outer_addition(vectors, output=None):
    """
    Corrected function to compute the outer addition of N K-dimensional vectors using broadcasting.
    This function is equivalent to the following code:
    ```
    result = torch.zeros((K1, K2, ..., KN))
    for idx1 in range(K1):
        for idx2 in range(K2):
            ...
            result[idx1, idx2, ..., idxn] = vectors[idx1] + vectors[idx2] + ... + vectors[idxn]
    ```
    However, it is much faster because it uses pre-computed sums and sums of squares.

    :param vectors: List of N vectors of shape (K1, K2, ..., KN)
    :param output: Optional output tensor
        if provided, must be of shape (K1, K2, ..., KN)
    :return: Tensor of shape (K1, K2, ..., KN)
    """

    # Assert all vectors are on the same device
    device = vectors[0].device
    assert all(
        v.device == device for v in vectors
    ), "All vectors must be on the same device"

    # Number of vectors (N) and dimensions (K)
    # N, K = vectors.shape
    N = len(vectors)
    Ks = [len(v) for v in vectors]
    if output is None:
        output = torch.zeros(Ks, dtype=vectors[0].dtype, device=vectors[0].device)
    else:
        assert output.shape == tuple(Ks), "Output tensor has incorrect shape"
        output.zero_()

    # Reshape each vector to have a unique non-singleton dimension
    for i in range(N):
        expanded_shape = [1] * N
        expanded_shape[i] = Ks[i]
        reshaped_vector = vectors[i].view(*expanded_shape)
        output += reshaped_vector

    return output


def compute_variances(X, Y):
    """
    Compute variances between all combinations of vectors in X and Y.
    This function is equivalent to the following code:
    ```
    variances = torch.zeros((X.size(0), Y.size(0)))
    for i in range(X.size(0)):
        for j in range(Y.size(0)):
            concatenated = torch.cat((X[i], Y[j]))
            variances[i, j] = torch.var(concatenated, unbiased=False)
    ```
    However, it is much faster because it uses pre-computed sums and sums of squares.


    :param X: Tensor of shape (N, K)
    :param Y: Tensor of shape (M, L)
    """

    # Compute sums and sums of squares for X
    sum_X = torch.sum(X, dim=1)
    sum_sq_X = torch.sum(X**2, dim=1)

    # Compute sums and sums of squares for Y
    sum_Y = torch.sum(Y, dim=1)
    sum_sq_Y = torch.sum(Y**2, dim=1)

    # Lengths of vectors in X and Y
    len_X = X.shape[1]
    len_Y = Y.shape[1]

    # Broadcasting sums and sum of squares for all combinations
    total_sum = sum_X.unsqueeze(1) + sum_Y.unsqueeze(0)
    total_sum_sq = sum_sq_X.unsqueeze(1) + sum_sq_Y.unsqueeze(0)
    total_len = len_X + len_Y

    # Compute variances
    mean = total_sum / total_len
    variances = total_sum_sq / total_len - mean**2

    return variances


def get_memory_capacity(device="cpu"):
    """
    Get the memory capacity of the specified device (CPU or GPU).

    :param device: 'cpu' or 'cuda'
    :return: Memory capacity in bytes
    """
    if torch.device(device).type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache to get more accurate readings
            gpu_id = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            return total_memory
        else:
            raise RuntimeError("CUDA is not available on this system.")
    else:
        # Using psutil to get system memory
        total_memory = psutil.virtual_memory().total
        return total_memory


def get_free_memory(device="cpu"):
    """
    Get the free memory of the specified device (CPU or GPU).

    :param device: 'cpu' or 'cuda'
    :return: Free memory in bytes
    """
    if torch.device(device).type == "cuda":
        if torch.cuda.is_available():
            # Get free memory in bytes
            torch.cuda.empty_cache()  # Clear cache to get more accurate readings
            gpu_id = torch.cuda.current_device()
            _, free_memory = torch.cuda.mem_get_info(gpu_id)
            return free_memory
        else:
            raise RuntimeError("CUDA is not available on this system.")
    else:
        # Using psutil to get system memory
        free_memory = psutil.virtual_memory().available
        return free_memory


def get_max_numel(dtype, memory_capacity=None, device="cpu"):
    """
    Compute the maximum number of elements that fit in specified memory.

    :param dtype: Data type of the tensor (e.g., torch.float32)
    :param memory_capacity: Memory capacity in bytes
    :param device: 'cpu' or 'cuda'
    :return: maximum number of elements that fit
    """

    # Get memory capacity
    if memory_capacity is None:
        memory_capacity = get_free_memory(device)

    # Calculate maximum number of elements that fit
    element_size = torch.tensor(
        [], dtype=dtype
    ).element_size()  # Size in bytes of one element
    max_numel = memory_capacity // element_size

    return max_numel


def check_matrix_fit_and_num_chunks(
    dimensions, dtype, memory_capacity=None, device="cpu"
):
    """
    Check if a tensor of given dimensions and data type fits in specified memory.
    If not, return chunk sizes that maximize the capacity, slicing only along the first dimension.

    :param dimensions: Tuple of dimensions for the tensor
    :param dtype: Data type of the tensor (e.g., torch.float32)
    :param memory_capacity: Memory capacity in bytes
    :param device: 'cpu' or 'cuda'
    :return: number of chunks along the first dimension
    """

    # Get memory capacity
    if memory_capacity is None:
        memory_capacity = get_memory_capacity(device)

    # Calculate total number of elements
    total_elements = 1
    for dim in dimensions:
        total_elements *= dim

    element_size = torch.tensor(
        [], dtype=dtype
    ).element_size()  # Size in bytes of one element
    total_size = total_elements * element_size  # Total memory required for the tensor

    if total_size <= memory_capacity:
        return 1

    # If doesn't fit, calculate chunk size for the first dimension
    other_dims_product = 1
    for dim in dimensions[1:]:
        other_dims_product *= dim

    max_first_dim_size = memory_capacity // (other_dims_product * element_size)
    if max_first_dim_size == 0:
        raise ValueError("Tensor does not fit in memory.")

    num_chunks = math.ceil(dimensions[0] / max_first_dim_size)

    return num_chunks


def convert_property_to_categorical(property):
    """
    Convert the properties to a categorical variable.

    :param property: List of properties for each rower.
        Shape: (num_rowers)
        dtype: Any

    :return: Tensor of categorical properties.
        Shape: (num_rowers)
        dtype: torch.long
    """

    unique_properties = set()
    for p in property:
        unique_properties.add(p)
    unique_properties = sorted(list(unique_properties))
    property = [unique_properties.index(p) for p in property]
    property = torch.tensor(property)
    return property


def extract_best_assignment(assignments_per_week, total_score):
    """
    Extract the best assignment for each outing.

    :param assignments_per_week: Tensor of assignments per week.
        shape: (num_outings, num_combinations, num_rowers)
    :param total_score: Tensor of total score for each assignment.
        shape: (num_combinations, num_combinations, ..., num_combinations) x num_outings

    :return: Tensor of best assignment per outing.
        shape: (num_outings, 1, num_rowers)

    """

    num_outings, num_combinations, num_rowers = assignments_per_week.shape

    # Find the top assignments
    # top_inds = torch.argsort(total_score.flatten(), descending=True)[0]
    top_idx = torch.argmax(total_score.flatten())

    top_idx = unravel_indices(top_idx, total_score.shape)

    # top_inds tells us for each outing the index of the top assignment
    top_assignment = torch.zeros(
        num_outings,
        1,
        num_rowers,
        dtype=torch.uint8,
        device=assignments_per_week.device,
    )
    for outing_idx, comb_idx in enumerate(top_idx):
        top_assignment[outing_idx] = assignments_per_week[outing_idx, comb_idx]

    return top_assignment


#############################################
# Per outing functions
#############################################
@torch.no_grad()
def get_no_overlap_inds(A, B):
    """
    Perform matrix multiplication of A and B in chunks.
    Return the indices of rows in A and columns in B that have no overlap.
    Overlap is defined as a non-zero value in the product of A and B.

    :param A: First matrix
        shape: (num_combinations_A, num_rowers)
    :param B: Second matrix
        shape: (num_combinations_B, num_rowers)
    :param chunk_sizes: Chunk sizes for the first dimension of A
    :return: indices of rows in A and columns in B that have no overlap
    """

    # check if the product of the two matrices fits in memory
    # if not, chunk the matrices and check for overlap in chunks
    num_chunks = check_matrix_fit_and_num_chunks(
        (A.shape[0], A.shape[1], B.shape[0]), dtype=A.dtype, device=A.device
    )

    # num_chunks = 1
    def multiply_and_find(A_chunk, B):
        # counts the number of double-assignments for each rower between the two boats
        assignment_count = torch.matmul(A_chunk, B.T)
        no_overlap_inds = torch.nonzero(assignment_count == 0)
        return no_overlap_inds

    # if the product fits in memory, check for overlap in one go
    if num_chunks == 1:
        return multiply_and_find(A, B)

    A_chunks = torch.chunk(A, num_chunks)

    # otherwise, chunk the matrices and check for overlap in chunks
    no_overlap_inds = []
    offset_idx = 0
    for A_chunk in tqdm.tqdm(A_chunks):
        # no_overlap_inds.append(multiply_and_find(A_chunk, B).tolist())
        chunk_inds = multiply_and_find(A_chunk, B)

        # add the chunk size to offset the indices
        chunk_inds[:, 0] += offset_idx
        offset_idx += A_chunk.shape[0]
        no_overlap_inds.append(chunk_inds)

    return torch.cat(no_overlap_inds)


@torch.no_grad()
def generate_binary_matrices(
    num_rowers,
    boat_sizes,
    device="cpu",
    max_num_combinations=NUM_MAX_COMBINATION_PER_BOAT,
):
    """
    Generate binary matrices for each combination of rowers in boats.

    :param num_rowers: Total number of rowers
    :param boat_sizes: List of boat sizes
    """
    per_boat_binary_matrices = []
    for boat_size in boat_sizes:
        # Precompute indices for combinations
        row_indices = []
        col_indices = []

        num_combinations = math.comb(num_rowers, boat_size)
        if num_combinations > max_num_combinations:
            M = torch.zeros((max_num_combinations, num_rowers), dtype=torch.bool)

            keep_indices = sample(
                torch.arange(num_combinations), k=max_num_combinations
            )
            keep_indices = keep_indices.sort().values
            i = 0
            for row, combination in enumerate(
                itertools.combinations(range(num_rowers), boat_size)
            ):
                if keep_indices[i] != row:
                    continue
                for col in combination:
                    row_indices.append(i)
                    col_indices.append(col)
                i += 1
                if i == max_num_combinations:
                    break

        else:
            M = torch.zeros((num_combinations, num_rowers), dtype=torch.bool)
            for row, combination in enumerate(
                itertools.combinations(range(num_rowers), boat_size)
            ):
                for col in combination:
                    row_indices.append(row)
                    col_indices.append(col)

        # Use advanced indexing to fill the matrix
        M[row_indices, col_indices] = 1
        per_boat_binary_matrices.append(M)
    return per_boat_binary_matrices


@torch.no_grad()
def eliminate_invalid_boats(
    binary_matrix, rower_sides, num_max_combinations=NUM_MAX_COMBINATION_PER_BOAT
):
    """
    Eliminate invalid boats from a binary matrix.

    Currently we consider a boat invalid if there are more rowers on one side than the other.
    We represent stroke as 1 and bow as -1 and 0 for no preference.

    :param binary_matrix: Binary matrix of rower combinations
        shape: (num_combinations, num_rowers)
    :return: Binary matrix with invalid boats eliminated
    """

    # gather the rower sides for each rower in each boat for each combination
    num_assigned_rowers = binary_matrix[0].sum()
    # assert each row has the same number of assigned rowers
    assert (binary_matrix.sum(dim=1) == num_assigned_rowers).all()
    assert len(rower_sides) == binary_matrix.shape[1]
    idx = binary_matrix.nonzero()[:, 1].view(len(binary_matrix), num_assigned_rowers)
    outings = rower_sides[idx]

    # Compute the offset between the number of stroke and bow seats
    offset = torch.sum(outings, dim=1).abs()
    # Determine the number of rowers that are both stroke and bow seat
    count_where_both = torch.sum(outings == 0, dim=1)

    # Eliminate invalid boats
    is_valid = count_where_both >= offset
    binary_matrix = binary_matrix[is_valid]

    if len(binary_matrix) > num_max_combinations:
        binary_matrix = sample(binary_matrix, k=num_max_combinations)

    return binary_matrix


@torch.no_grad()
def generate_valid_assignments(
    single_boat_bin_matrices, num_max_combinations=NUM_MAX_PAIRWISE_COMBINATION
):
    """
    Generate valid combinations of rowers across multiple boats on a single outing

    :param matrices: List of binary matrices, each representing combinations for a boat.
        shape: List[
            Tensor(num_combinations_1, num_rowers),
            Tensor(num_combinations_2, num_rowers),
            ...
            Tensor(num_combinations_n, num_rowers),
        ]
    :return: Tensor of valid combinations across all boats.
    """
    assert len(single_boat_bin_matrices) > 0, "Must have at least one boat"
    assert all(
        m.shape[1] == single_boat_bin_matrices[0].shape[1]
        for m in single_boat_bin_matrices
    ), "All matrices must have the same number of rowers"

    assignments = single_boat_bin_matrices[0]
    for boat_ind, boat_B in enumerate(single_boat_bin_matrices[1:], start=2):
        no_overlap_inds = get_no_overlap_inds(assignments, boat_B)

        if len(no_overlap_inds) > num_max_combinations:
            no_overlap_inds = sample(no_overlap_inds, k=num_max_combinations)

        A_inds, B_inds = no_overlap_inds.T

        # update boat_A to be the combination of boat_A and boat_B with no overlap
        assignments = assignments[A_inds] + boat_B[B_inds] * boat_ind
    return assignments


@torch.no_grad()
def generate_canonic_valid_assignments_per_outing(
    outing_availabilities, boat_sizes, properties
):
    """
    Generate valid combinations of rowers across multiple boats on a single outing.

    :param outing_availabilities: Tensor of rower availabilities for a single outing.
        shape: (num_rowers)
        dtype: torch.bool
    :param boat_sizes: List of boat sizes
    :param properties: dict of Tensors of properties.
    :return: Tensor of valid combinations across all boats in the canonic form.
        shape: (num_combinations, num_rowers)
        dtype: torch.uint8
    """
    num_available_rowers = outing_availabilities.sum().item()
    if num_available_rowers < sum(boat_sizes):
        raise ValueError(
            f"Not enough rowers available for the outing: rowers ({num_available_rowers}) < seats ({sum(boat_sizes)})"
        )
    num_rowers = len(outing_availabilities)

    # Generate binary matrices for each boat
    binary_matrices = generate_binary_matrices(
        num_available_rowers, boat_sizes, device=outing_availabilities.device
    )

    # Move binary matrices to the same device as the availabilities
    binary_matrices = [m.to(outing_availabilities.device) for m in binary_matrices]

    # Eliminate invalid boats from each binary matrix
    properties = {k: v[outing_availabilities] for k, v in properties.items()}
    for i, binary_matrix in enumerate(binary_matrices):
        binary_matrices[i] = eliminate_invalid_boats(binary_matrix, properties["side"])

    # Generate valid combinations across all boats
    valid_assignments = generate_valid_assignments(binary_matrices)
    num_combinations = len(valid_assignments)
    # valid_assignments has the shape of (num_combinations, num_available_rowers)

    # Convert to canonic form with shape (num_combinations, num_rowers)
    canonic_valid_assignments = torch.zeros(
        (num_combinations, num_rowers),
        dtype=outing_availabilities.dtype,
        device=outing_availabilities.device,
    )
    canonic_valid_assignments[:, outing_availabilities] = valid_assignments
    return canonic_valid_assignments


#############################################
# Per week functions
#############################################
def generate_canonic_valid_assignments(
    avail_per_week,
    boat_sizes_per_week,
    properties,
    num_combinations=None,
    callback=lambda x: None,
):
    num_outings = len(avail_per_week)
    if num_combinations is None:
        # this is based off on experimental data
        # to maximize memory usage (look at benchmark_peak_performance.pdf)
        num_combinations = {
            2: 4301,
            3: 200,
            4: 61,
            5: 27,
            6: 15,
            7: 10,
            8: 8,
            9: 6,
            10: 5,
            11: 4,
            12: 3,
            13: 3,
            14: 3,
            15: 3,
            16: 3,
            17: 3,
            18: 3,
            19: 2,
            20: 2,
            21: 2,
            22: 2,
            23: 2,
            24: 2,
            25: 2,
            26: 2,
            27: 2,
            28: 2,
            29: 2,
        }[num_outings]

    valid_assignments_per_week = []
    assignments_per_week = []
    start = time.time()
    for outing_avail, boat_sizes_per_outing in zip(avail_per_week, boat_sizes_per_week):
        if (time.time() - start) > 1:
            start = time.time()
            callback(
                f"Generating valid assignments: {len(valid_assignments_per_week)}/{num_outings}"
            )

        valid_assignments_per_outing = generate_canonic_valid_assignments_per_outing(
            outing_avail, boat_sizes_per_outing, properties
        )

        # randomly sample the valid combinations
        selected_combinations = sample(valid_assignments_per_outing, k=num_combinations)

        assignments_per_week.append(selected_combinations)
        valid_assignments_per_week.append(valid_assignments_per_outing)

    assignments_per_week = torch.stack(assignments_per_week)
    return valid_assignments_per_week, assignments_per_week


def evaluate_skill_variance(assignments_per_week, skill_levels, dtype=torch.float16):
    """
    This relies on the notion that the skill levels entered are not categorical
    but integer values (or can be mapped to ordered categories, e.g. M1 > M2 > M3 ... )

    :param assignments_per_week: Tensor of assignments per week.
        shape: (num_outings, num_combinations, num_rowers)

    :param skill_levels: Tensor of skill levels for each rower.
        shape: (num_rowers,)

    :return: Tensor of variance for each combination in each outing.
        shape: (num_combinations, num_combinations, ..., num_combinations) x num_outings
    """

    # assert that the number of assigned rowers is the same for each outing
    for outing_idx in range(len(assignments_per_week)):
        num_assigned_rowers = assignments_per_week[outing_idx][0].sum()
        assert (
            assignments_per_week[outing_idx].sum(dim=1) == num_assigned_rowers
        ).all()

    num_outings, num_combinations, num_rowers = assignments_per_week.shape
    max_num_boats = assignments_per_week.max().item()
    outing_variance = torch.zeros(
        num_outings, num_combinations, device=assignments_per_week.device, dtype=dtype
    )
    for boat_idx in range(max_num_boats):
        boat_assignment = assignments_per_week == boat_idx + 1
        # we use binary masking
        X = skill_levels * boat_assignment

        # but we need to make sure that we don't include the rowers that are not assigned
        X_sum = X.sum(dim=2)
        X_len = boat_assignment.sum(dim=2)
        X_mean = X_sum / X_len

        boat_variance = ((X - X_mean.unsqueeze_(2)) * boat_assignment) ** 2
        boat_variance = boat_variance.sum(dim=2)

        # we use the unbiased variance since the sample size is small
        boat_variance /= torch.clamp(X_len - 1, min=1)

        outing_variance += boat_variance

    # now we need to compute the variance between the outings across the week
    week_variance = generalized_outer_addition(outing_variance)
    return week_variance


def evaluate_num_preferred_outings(
    assignments_per_week, num_preferred_outings, dtype=torch.long
):
    # assert that the number of assigned rowers is the same for each outing
    for outing_idx in range(len(assignments_per_week)):
        num_assigned_rowers = assignments_per_week[outing_idx, 0].sum()
        assert (
            assignments_per_week[outing_idx].sum(dim=1) == num_assigned_rowers
        ).all()

    assignments_per_week = assignments_per_week > 0

    num_outings, num_combinations, num_rowers = assignments_per_week.shape

    # just to pin memory and reuse the output tensor
    num_assignment_per_rower = torch.zeros(
        [num_combinations] * num_outings,
        device=assignments_per_week.device,
        dtype=dtype,
    )

    week_over_assignment = torch.zeros(
        [num_combinations] * num_outings,
        device=assignments_per_week.device,
        dtype=dtype,
    )

    for rower_idx in range(num_rowers):
        num_assignment_per_rower = generalized_outer_addition(
            assignments_per_week[:, :, rower_idx], output=num_assignment_per_rower
        )
        num_preferred_outings_per_rower = num_preferred_outings[rower_idx]
        assignment_diff = num_assignment_per_rower - num_preferred_outings_per_rower
        over_assignment = assignment_diff.clamp_(min=0)
        week_over_assignment += over_assignment

    return week_over_assignment


def evaluate_assignments_per_week(
    assignments_per_week, properties, weights, return_stats=False
):
    """
    Evaluate the assignments per week.

    :param assignments_per_week: Tensor of num_outings different assignments for the week.
        Shape: (num_outings, num_combinations, num_rowers)
        dtype: torch.uint8
    :param properties: dict of Tensors of properties.
        Shape: {property_name: Tensor(num_rowers)}
        dtype: torch.long
    :param weights: dict of weights for each property.
        Shape: {property_name: float}
    :param return_stats: Whether to return the stats for each property.

    :return: Total score for the week.
        Shape: (num_combinations, num_combinations, ..., num_combinations) x num_outings
    :return: Stats for each weight category.
    """

    # Compute variance of skill levels
    week_variance = evaluate_skill_variance(
        assignments_per_week, properties["skill_level"]
    )

    # Compute number of preferred outings
    week_num_preferred_outings = evaluate_num_preferred_outings(
        assignments_per_week, properties["num_preferred_outings"]
    )

    # Compute total score
    total_score = (
        weights["skill variance"] * week_variance
        + weights["over assignment"] * week_num_preferred_outings
    )

    if return_stats:
        stats = {
            "values": {
                "skill variance": week_variance,
                "over assignment": week_num_preferred_outings,
            },
            "weights": weights,
            "total": total_score,
        }
        return total_score, stats

    return total_score


def permute_top_assignments(
    valid_assignments,
    assignments_per_week,
    total_scores,
    num_permutations=10,
    randomize_permutations=True,
):
    """
    Permute the top assignments for the week.
    """
    num_outings, num_combinations, num_rowers = assignments_per_week.shape

    assert len(valid_assignments) == num_outings, "Must have the same number of outings"
    assert (
        len(assignments_per_week) == num_outings
    ), "Must have the same number of outings"
    if any(m.ndim != 2 for m in valid_assignments):
        raise ValueError("All outing assignments have to be 2D for every outing")
    if any(m.shape[1] != num_rowers for m in valid_assignments):
        raise ValueError(
            "All outing assignments have to have the same number of rowers"
        )
    if any((m.sum(dim=1) != m[0].sum()).any() for m in valid_assignments):
        raise ValueError(
            f"In each combination of every outing,\
                          the number of rowers assigned must be the same."
        )

    # assert all(
    #     m.ndim == 2
    #     for m in valid_assignments
    # ), f"All matrices must have the same number of dim: {[m.shape for m in valid_assignments]}"
    # assert all(
    #     m.shape[1] == num_rowers
    #     for m in valid_assignments
    # ), "All matrices must have the same number of rowers"
    # for outing_idx in range(len(valid_assignments)):
    #     assert (valid_assignments[outing_idx].sum() == valid_assignments[outing_idx][0].sum()).all(),\
    #         "Combinations must have the same number of rowers assigned in an outing"

    # assert that the number of assigned rowers is the same for each outing
    for outing_idx in range(len(assignments_per_week)):
        num_assigned_rowers = assignments_per_week[outing_idx, 0].sum()
        assert (
            assignments_per_week[outing_idx].sum(dim=1) == num_assigned_rowers
        ).all()

    best_assignment = extract_best_assignment(assignments_per_week, total_scores)

    # in the permutations we fix all outings except the outing we are permuting
    permuted_assignment = best_assignment.repeat(1, num_permutations + 1, 1)
    for outing_idx in range(len(assignments_per_week)):
        # just copy the best assignment num_permutations times
        if randomize_permutations:
            # we need to make sure that the best assignment is included
            permuted_assignment[outing_idx, 1:] = sample(
                valid_assignments[outing_idx], k=num_permutations
            )
        else:
            permuted_assignment[outing_idx, 1:] = valid_assignments[outing_idx][
                :num_permutations
            ]
    return permuted_assignment


def iterative_permutation_improvement(
    valid_assignments,
    assignments_per_week,
    properties,
    weights,
    num_iterations=10,
    callback=lambda x: None,
):
    total_scores = evaluate_assignments_per_week(
        assignments_per_week, properties, weights
    )
    best_assignment = extract_best_assignment(assignments_per_week, total_scores)
    best_score = total_scores.max().item()

    num_outings, num_combinations, num_rowers = assignments_per_week.shape

    start = time.time()
    for it_idx in range(num_iterations):
        permutations = permute_top_assignments(
            valid_assignments=valid_assignments,
            assignments_per_week=assignments_per_week,
            total_scores=total_scores,
            num_permutations=num_combinations - 1,
        )
        permuted_scores = evaluate_assignments_per_week(
            permutations, properties, weights
        )

        if (time.time() - start) > 1:
            start = time.time()
            callback(
                f"Running iterative improvement: {int(it_idx/num_iterations*100)}%"
            )

        if permuted_scores.max() > best_score:
            logging.info(f"iteration {it_idx}: {permuted_scores.max()}")
            total_scores = permuted_scores
            best_score = permuted_scores.max().item()
            best_assignment = extract_best_assignment(permutations, permuted_scores)
            assignments_per_week = permutations

    return best_assignment, best_score


def clean_dataframes(avail_df, prop_df):
    """
    Removes non-rowers from the property dataframe
    and removes rowers who didn't sign up for the week at all
    to save compute and memory.
    """

    # remove non-rowers from the property dataframe
    prop_df["side"] = prop_df["side"].str.lower()

    rower_prop_df = prop_df[
        prop_df["side"].isin(["s", "b", "both", "both (s)", "both (b)"])
    ]
    # make contiguous
    rower_prop_df = rower_prop_df.reset_index(drop=True)

    # merge the two dataframes to drop those who didn't sign up for the week at all
    merged_df = pd.merge(rower_prop_df, avail_df, on="name")

    # split the dataframe into rower properties and availabilities excluding the name column
    rower_avail_df = merged_df.drop(rower_prop_df.columns.drop("name"), axis=1)
    rower_prop_df = merged_df.drop(avail_df.columns.drop("name"), axis=1)

    rower_avail_df = rower_avail_df.set_index("name")
    rower_prop_df = rower_prop_df.set_index("name")

    return rower_avail_df, rower_prop_df


def convert_df_to_tensors(avail_df, prop_df, boat_sizes_per_week, device):
    """
    Converts a dataframe of rowers to a list of lists of rowers
    """

    # convert the rower availabilities to tensors
    rower_availabilities = avail_df.astype(bool).to_numpy()
    rower_availabilities = torch.tensor(rower_availabilities).to(device)

    # when there are no boats available (i.e. [])
    # we drop the outing from the week
    outing_mask = [len(size_per_outing) > 0 for size_per_outing in boat_sizes_per_week]
    rower_availabilities = rower_availabilities[:, outing_mask]
    # transpose the rower availabilities to have the shape of (num_outings, num_rowers)
    rower_availabilities = rower_availabilities.T.contiguous()
    relevant_boat_sizes = []
    for outing_idx, size_per_outing in enumerate(boat_sizes_per_week):
        if outing_mask[outing_idx]:
            relevant_boat_sizes.append(size_per_outing)

    # convert the rower properties to tensors
    side = prop_df["side"]

    # replace "both (s)", "both (b)" with "0"
    side = side.replace(["both (s)", "both (b)", "both"], 0)
    # replace "s" with "1"
    side = side.replace("s", 1)
    # replace "b" with "-1"
    side = side.replace("b", -1)
    side = torch.tensor(side.to_numpy().astype(int))

    skill_level = convert_property_to_categorical(prop_df["skill_level"])

    # replace nan with 3
    prop_df["num_preferred_outings"].fillna(3, inplace=True)

    num_preferred_outings = torch.tensor(
        prop_df["num_preferred_outings"].to_numpy().astype(int)
    )

    rower_properties = {
        "side": side.to(device),
        "skill_level": skill_level.to(device),
        "num_preferred_outings": num_preferred_outings.to(device),
    }

    return rower_availabilities, rower_properties, relevant_boat_sizes


def convert_results_to_json(
    best_assignment,
    avail_df,
    prop_df,
    boat_sizes_per_week,
):
    rowers = []
    # unset the name index
    prop_df = prop_df.reset_index()
    avail_df = avail_df.reset_index()

    # convert the rower properties to dicts of rowers
    for rower_idx, rower in prop_df.iterrows():
        rowers.append(rower.to_dict())

    # convert the assignments to the following format:
    # {
    #     <outing_1 name>: [
    #       [<rower_id>, <rower_id>, ...], <--- boat 1
    #       [<rower_id>, <rower_id>, ...], <--- boat 2
    #       ...
    #       [<rower_id>, <rower_id>, ...], <--- boat n
    #       [<rower_id>, <rower_id>, ...], <--- reserves
    #     ],
    #     <outing_2 name>: [
    #       [<rower_id>, <rower_id>, ...], <--- boat 1
    #       [<rower_id>, <rower_id>, ...], <--- boat 2
    #       ...
    #       [<rower_id>, <rower_id>, ...], <--- boat n
    #       [<rower_id>, <rower_id>, ...], <--- reserves
    #     ],
    #     ...
    #     <outing_n name>: [
    #       ...
    #     ]
    # }

    # Normalize the best_assignments to include the outings with no boats
    num_rowers = len(rowers)
    normalized_best_assignment = torch.zeros(
        len(boat_sizes_per_week),
        num_rowers,
        dtype=torch.uint8,
    )
    i = 0
    for outing_idx, boat_sizes in enumerate(boat_sizes_per_week):
        if len(boat_sizes) > 0:
            normalized_best_assignment[outing_idx] = best_assignment[i]
            i += 1

    best_assignment = normalized_best_assignment

    assignments = {}
    outing_names = avail_df.columns[1:]
    for outing_idx, outing_name in enumerate(outing_names):
        outing_assignments = []
        num_boats = best_assignment[outing_idx].max().item()
        for boat_idx in range(num_boats):
            boat_assignment = best_assignment[outing_idx] == boat_idx + 1
            boat_assignment = boat_assignment.nonzero().flatten().tolist()
            # sort the rowers wrt to name
            boat_assignment = sorted(boat_assignment, key=lambda x: rowers[x]["name"])
            # sort the rowers wrt to the skill level
            boat_assignment = sorted(
                boat_assignment, key=lambda x: rowers[x]["skill_level"]
            )
            outing_assignments.append(boat_assignment)

        # add the reserves to the end of the list
        reserves = (best_assignment[outing_idx] == 0).nonzero().flatten().tolist()

        # remove those who are not available
        reserves = [r for r in reserves if avail_df[outing_name][r]]

        # sort the reserves wrt to the name
        reserves = sorted(reserves, key=lambda x: rowers[x]["name"])
        # sort the reserves wrt to the skill level
        reserves = sorted(reserves, key=lambda x: rowers[x]["skill_level"])

        outing_assignments.append(reserves)
        assignments[outing_name] = outing_assignments

    # compute the stats
    results = {
        "assignments": assignments,
        "rowers": rowers,
    }
    return results


def solve_week(
    avail_df,
    prop_df,
    weights,
    boat_sizes,
    num_iterations=200,
    device=None,
    callback=lambda x: None,
):
    logging.info("Week solver started")
    logging.info(f"Weights: {weights}")
    logging.info(f"Boat sizes: {boat_sizes}")
    logging.info(f"Number of iterations: {num_iterations}")

    if device is None:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
    logging.info(f"Device: {device}")

    callback("Converting to tensors")
    logging.info("Cleaning the dataframes")
    avail_df, prop_df = clean_dataframes(avail_df, prop_df)

    logging.info("Converting the availabilities and props to tensors")
    availabilities, properties, relevant_boat_sizes = convert_df_to_tensors(
        avail_df=avail_df,
        prop_df=prop_df,
        boat_sizes_per_week=boat_sizes,
        device=device,
    )
    weights = {k: torch.tensor(v, device=device) for k, v in weights.items()}

    callback("Generating valid assignments")
    logging.info("Generating valid assignments per outing")
    valid_assignments, week_assignment = generate_canonic_valid_assignments(
        availabilities, relevant_boat_sizes, properties, callback=callback
    )
    num_outings, num_combinations, num_rowers = week_assignment.shape
    logging.info(f"Number of outings: {num_outings}")
    logging.info(f"Number of combinations: {num_combinations}")
    logging.info(f"Number of rowers: {num_rowers}")

    callback("Evaluating the assignments")
    logging.info("Evaluating the assignments per week")
    total_scores = evaluate_assignments_per_week(week_assignment, properties, weights)
    num_week_evaluations = total_scores.numel()
    logging.info(f"Inital score: {total_scores.max().item()}")

    callback("Permuting the top assignments")
    logging.info("Generating permutations for each outing")
    # re-generate valid combinations for each outing for permutation
    valid_assignments, _ = generate_canonic_valid_assignments(
        availabilities, relevant_boat_sizes, properties
    )

    callback("Running iterative improvement")
    logging.info("Beginning iterative permutation improvement")
    # improve the assignments by permuting the top assignments
    best_assignment, best_score = iterative_permutation_improvement(
        valid_assignments,
        week_assignment,
        properties,
        weights,
        num_iterations,
        callback=callback,
    )

    logging.info(f"Final score: {best_score}")
    num_week_evaluations += (num_combinations) ** num_outings * num_iterations
    logging.info(f"Number of evaluations of weekly assignemnts: {num_week_evaluations}")

    callback("Converting results to json")
    logging.info("Converting the results to json format")
    results = convert_results_to_json(
        best_assignment,
        avail_df,
        prop_df,
        boat_sizes_per_week=boat_sizes,
    )

    logging.info("Adding statistics to the results")
    _, stats = evaluate_assignments_per_week(
        best_assignment, properties, weights, return_stats=True
    )
    for k, v in stats["weights"].items():
        stats["weights"][k] = v.cpu().item()
    for k, v in stats["values"].items():
        stats["values"][k] = v.cpu().item()
        logging.info(f"{k}: {v.cpu().item()}")
    stats["n"] = num_week_evaluations
    stats["final_score"] = best_score
    results["stats"] = stats

    logging.info("Week solver finished")
    return results
