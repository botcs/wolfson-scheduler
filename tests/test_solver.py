import torch
import unittest
import math
from unittest.mock import patch

from solver import (
    unravel_indices,
    generalized_outer_addition,
    compute_variances,
    get_max_numel,
    check_matrix_fit_and_num_chunks,
    convert_property_to_categorical,
    extract_best_assignment,
    get_no_overlap_inds,
    generate_binary_matrices,
    eliminate_invalid_boats,
    generate_valid_assignments,
    evaluate_skill_variance,
    evaluate_num_preferred_outings,
    evaluate_assignments_per_week,
    permute_top_assignments,
)

class TestUnravelIndices(unittest.TestCase):

    def test_simple_case(self):
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        shape = (2, 3)
        expected_result = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        result = unravel_indices(indices, shape)
        self.assertTrue(torch.equal(result, expected_result))

    def test_single_dimension(self):
        indices = torch.tensor([0, 1, 2, 3])
        shape = (4,)
        expected_result = torch.tensor([[0], [1], [2], [3]])
        result = unravel_indices(indices, shape)
        self.assertTrue(torch.equal(result, expected_result))

    def test_multi_dimension(self):
        indices = torch.tensor([0, 1, 5, 11])
        shape = (2, 3, 2)
        expected_result = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 2, 1], [1, 2, 1]])
        result = unravel_indices(indices, shape)
        self.assertTrue(torch.equal(result, expected_result))

    def test_edge_cases(self):
        indices = torch.tensor([0])
        shape = (1, 1, 1)
        expected_result = torch.tensor([[0, 0, 0]])
        result = unravel_indices(indices, shape)
        self.assertTrue(torch.equal(result, expected_result))

    def test_output_type_and_shape(self):
        indices = torch.tensor([3, 7])
        shape = (2, 4)
        result = unravel_indices(indices, shape)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 2))


class TestGeneralizedOuterAddition(unittest.TestCase):

    def test_correct_calculation(self):
        vectors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        expected_result = torch.tensor([[4, 5], [5, 6]])
        result = generalized_outer_addition(vectors)
        self.assertTrue(torch.equal(result, expected_result))

    def test_different_vector_sizes(self):
        vectors = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        expected_result = torch.tensor([[4, 5, 6], [5, 6, 7]])
        result = generalized_outer_addition(vectors)
        self.assertTrue(torch.equal(result, expected_result))

    def test_with_output_tensor(self):
        vectors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        output = torch.empty((2, 2))
        expected_result = torch.tensor([[4, 5], [5, 6]])
        result = generalized_outer_addition(vectors, output)
        self.assertTrue(torch.equal(result, expected_result))

    def test_error_with_incorrect_output_shape(self):
        vectors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        output = torch.empty((3, 3))
        with self.assertRaises(AssertionError):
            generalized_outer_addition(vectors, output)

    def test_type_and_device_consistency(self):
        vectors = [torch.tensor([1., 2.], device="cuda"), torch.tensor([3., 4.], device="cuda")]
        result = generalized_outer_addition(vectors)
        self.assertTrue(result.dtype == torch.float32)
        self.assertTrue(result.device.type == "cuda")


class TestComputeVariances(unittest.TestCase):
    def test_variances(self):
        # Create sample data
        torch.manual_seed(0)  # For reproducibility
        X = torch.rand(3, 7)
        Y = torch.rand(4, 5)

        # Expected variances computed by manual concatenation
        expected_variances = torch.zeros((X.size(0), Y.size(0)))
        for i in range(X.size(0)):
            for j in range(Y.size(0)):
                concatenated = torch.cat((X[i], Y[j]))
                expected_variances[i, j] = torch.var(concatenated, unbiased=False)

        # Variances computed by the function
        actual_variances = compute_variances(X, Y)

        # Assert equality (within a tolerance to account for floating-point errors)
        self.assertTrue(torch.allclose(expected_variances, actual_variances, atol=1e-6))


class TestGetMaxNumel(unittest.TestCase):

    @patch('solver.get_free_memory')
    def test_with_different_dtypes(self, mock_get_free_memory):
        mock_get_free_memory.return_value = 1024  # Mock 1024 bytes of free memory
        dtypes = [torch.float32, torch.int32, torch.float64]
        for dtype in dtypes:
            element_size = torch.tensor([], dtype=dtype).element_size()
            expected_result = 1024 // element_size
            result = get_max_numel(dtype)
            self.assertEqual(result, expected_result)

    @patch('solver.get_free_memory')
    def test_without_specified_memory_capacity(self, mock_get_free_memory):
        mock_get_free_memory.return_value = 2048  # Mock 2048 bytes of free memory
        dtype = torch.float32
        element_size = torch.tensor([], dtype=dtype).element_size()
        expected_result = 2048 // element_size
        result = get_max_numel(dtype)
        self.assertEqual(result, expected_result)

    def test_with_specified_memory_capacity(self):
        dtype = torch.float32
        memory_capacity = 4096  # Specify 4096 bytes of memory
        element_size = torch.tensor([], dtype=dtype).element_size()
        expected_result = 4096 // element_size
        result = get_max_numel(dtype, memory_capacity)
        self.assertEqual(result, expected_result)


class TestCheckMatrixFitAndNumChunks(unittest.TestCase):
    def test_tensor_fits_memory(self):
        dimensions = (10, 10, 10)
        dtype = torch.float32
        memory_capacity = 40000  # Set a capacity that's more than enough
        self.assertEqual(check_matrix_fit_and_num_chunks(dimensions, dtype, memory_capacity), 1)

    def test_tensor_exceeds_memory(self):
        dimensions = (100, 100, 100)
        dtype = torch.float32
        memory_capacity = 1000  # Set a capacity that's too small
        self.assertRaises(ValueError, check_matrix_fit_and_num_chunks, dimensions, dtype, memory_capacity)

    def test_different_data_types(self):
        dimensions = (100, 100)
        memory_capacity = 100000
        for dtype in [torch.float32, torch.int32, torch.float64]:
            self.assertIsInstance(check_matrix_fit_and_num_chunks(dimensions, dtype, memory_capacity), int)

    def test_various_dimensions(self):
        dtype = torch.float32
        memory_capacity = 10000
        test_dimensions = [
            (100, 20, 5),
            (50, 40, 30),
            (200, 10, 10)
        ]
        for dimensions in test_dimensions:
            self.assertIsInstance(check_matrix_fit_and_num_chunks(dimensions, dtype, memory_capacity), int)

    def test_without_specified_memory_capacity(self):
        dimensions = (10, 10, 10)
        dtype = torch.float32
        self.assertIsInstance(check_matrix_fit_and_num_chunks(dimensions, dtype), int)


class TestConvertPropertyToCategorical(unittest.TestCase):

    def test_correct_conversion(self):
        property_list = ["red", "blue", "red"]
        expected_result = torch.tensor([1, 0, 1])
        result = convert_property_to_categorical(property_list)
        self.assertTrue(torch.equal(result, expected_result))

    def test_empty_input(self):
        property_list = []
        expected_result = torch.tensor([])
        result = convert_property_to_categorical(property_list)
        self.assertTrue(torch.equal(result, expected_result))

    def test_mixed_values(self):
        property_list = ["apple", "banana", "apple", "cherry"]
        expected_result = torch.tensor([0, 1, 0, 2])
        result = convert_property_to_categorical(property_list)
        self.assertTrue(torch.equal(result, expected_result))

    def test_consistency_in_indexing(self):
        property_list = ["dog", "cat", "bird", "cat"]
        expected_result = torch.tensor([2, 1, 0, 1])
        result = convert_property_to_categorical(property_list)
        self.assertTrue(torch.equal(result, expected_result))

    def test_output_type_and_shape(self):
        property_list = ["one", "two", "three"]
        result = convert_property_to_categorical(property_list)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.int64)
        self.assertEqual(result.shape, (3,))


class TestExtractBestAssignment(unittest.TestCase):

    def test_valid_inputs(self):
        # Mock data
        assignments_per_week = torch.randint(0, 2, (3, 4, 5), dtype=torch.uint8)
        total_score = torch.rand(4, 4, 4)  # Mock score tensor for 3 outings

        # Expected output shape
        expected_shape = (3, 1, 5)

        result = extract_best_assignment(assignments_per_week, total_score)
        self.assertEqual(result.shape, expected_shape)

    def test_edge_case_single_outing(self):
        assignments_per_week = torch.randint(0, 2, (1, 4, 5), dtype=torch.uint8)
        total_score = torch.rand(4,)

        expected_shape = (1, 1, 5)

        result = extract_best_assignment(assignments_per_week, total_score)
        self.assertEqual(result.shape, expected_shape)

    def test_output_type(self):
        assignments_per_week = torch.randint(0, 2, (3, 4, 5), dtype=torch.uint8)
        total_score = torch.rand(4, 4, 4)

        result = extract_best_assignment(assignments_per_week, total_score)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(result.dtype, torch.uint8)

    def test_correctness_of_assignment_extraction(self):
        # Mock data for 3 outings with 4 combinations each
        assignments_per_week = torch.tensor([
            [[0, 0], [0, 1], [1, 0], [1, 1]],  # Outing 1
            [[0, 0], [0, 1], [1, 0], [1, 1]],  # Outing 2
            [[0, 0], [0, 1], [1, 0], [1, 1]]   # Outing 3
        ], dtype=torch.uint8)

        # Mock total scores where the best scores are known
        # Assuming the best scores are for the combinations [1, 0, 3] for outings [1, 2, 3]
        total_score = torch.zeros((4, 4, 4))
        total_score[1, 0, 3] = 1  # Highest score

        # Expected best assignments for each outing
        expected_assignments = torch.tensor([
            [[0, 1]],  # Outing 1
            [[0, 0]],  # Outing 2
            [[1, 1]]   # Outing 3
        ], dtype=torch.uint8)  # Add dimension to match the expected output shape

        result = extract_best_assignment(assignments_per_week, total_score)
        self.assertTrue(torch.equal(result, expected_assignments))


class TestGetNoOverlapInds(unittest.TestCase):

    def test_no_overlap(self):
        A = torch.tensor([[1, 0], [0, 1]])
        B = torch.tensor([[0, 1], [1, 0]])
        expected_result = torch.tensor([[0, 0], [1, 1]])
        result = get_no_overlap_inds(A, B)
        self.assertTrue(torch.equal(result, expected_result))

    def test_partial_overlap(self):
        A = torch.tensor([[1, 1], [0, 1]])
        B = torch.tensor([[1, 0], [0, 1]])
        expected_result = torch.tensor([[1, 0]])
        result = get_no_overlap_inds(A, B)
        self.assertTrue(torch.equal(result, expected_result))

    def test_complete_overlap(self):
        A = torch.tensor([[1, 1], [1, 1]])
        B = torch.tensor([[1, 1], [1, 1]])
        expected_result = torch.empty((0, 2), dtype=torch.int64)
        result = get_no_overlap_inds(A, B)
        self.assertTrue(torch.equal(result, expected_result))

    def test_different_sizes(self):
        A = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])
        B = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
        expected_result = torch.tensor([[1, 2]])
        result = get_no_overlap_inds(A, B)
        self.assertTrue(torch.equal(result, expected_result))


class TestGenerateBinaryMatrices(unittest.TestCase):

    def test_correct_matrix_generation(self):
        num_rowers = 4
        boat_sizes = [2, 3]
        expected_combinations = [math.comb(num_rowers, boat_size) for boat_size in boat_sizes]

        result_matrices = generate_binary_matrices(num_rowers, boat_sizes)

        for i, M in enumerate(result_matrices):
            self.assertEqual(M.shape[0], expected_combinations[i])  # Correct number of combinations
            self.assertEqual(M.shape[1], num_rowers)  # Correct number of columns
            self.assertTrue(torch.all((M.sum(axis=1) == boat_sizes[i]).logical_or(M.sum(axis=1) == 0)))  # Correct boat sizes

    def test_different_rower_and_boat_sizes(self):
        num_rowers = 5
        boat_sizes = [1, 4]
        result_matrices = generate_binary_matrices(num_rowers, boat_sizes)

        for M, boat_size in zip(result_matrices, boat_sizes):
            self.assertEqual(M.shape, (math.comb(num_rowers, boat_size), num_rowers))

    def test_output_type(self):
        num_rowers = 3
        boat_sizes = [2]
        result_matrices = generate_binary_matrices(num_rowers, boat_sizes)

        for M in result_matrices:
            self.assertIsInstance(M, torch.Tensor)
            self.assertTrue(M.dtype, torch.bool)



class TestEliminateInvalidBoats(unittest.TestCase):

    def test_no_elimination_of_valid_boats(self):
        binary_matrix = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        rower_sides = torch.tensor([1, -1, 0])  # Stroke, Bow, No preference
        expected_result = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])  # Eliminate [1, 1, 0] combination
        result = eliminate_invalid_boats(binary_matrix, rower_sides)
        self.assertTrue(torch.equal(result, expected_result))

    def test_elimination_of_invalid_boats(self):
        binary_matrix = torch.tensor([[1, 1, 0], [1, 0, 1]])
        rower_sides = torch.tensor([1, 0, 1])  # Stroke, No preference, Stroke
        # Eliminate [1, 0, 1] combination because of two stroke siders
        expected_result = torch.tensor([[1, 1, 0]])
        result = eliminate_invalid_boats(binary_matrix, rower_sides)
        self.assertTrue(torch.equal(result, expected_result))

    def test_combination_limit(self):
        binary_matrix = torch.tensor([[1, 0, 1], [1, 0, 1], [0, 1, 1]])
        rower_sides = torch.tensor([1, -1, 0])  # Stroke, Bow
        num_max_combinations = 2
        result = eliminate_invalid_boats(binary_matrix, rower_sides, num_max_combinations)
        self.assertLessEqual(len(result), num_max_combinations)

    def test_output_type_and_shape(self):
        binary_matrix = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        rower_sides = torch.tensor([1, -1, 0])
        result = eliminate_invalid_boats(binary_matrix, rower_sides)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dim(), 2)

class TestGenerateValidCombinations(unittest.TestCase):

    def test_valid_combinations(self):
        A = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0]])
        B = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]])
        C = torch.tensor([[0, 0, 0, 0, 1, 1]])
        result = generate_valid_assignments([A, B, C])
        expected_result = torch.tensor([[2, 1, 1, 2, 3, 3]])
        self.assertTrue(torch.equal(result, expected_result))

    def test_combination_limit(self):
        A = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 ,0], [0, 0, 0, 1]])
        B = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 ,0], [0, 0, 0, 1]])
        C = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 ,0], [0, 0, 0, 1]])
        num_max_combinations = 2
        result = generate_valid_assignments([A, B, C], num_max_combinations)
        self.assertLessEqual(len(result), num_max_combinations)

    def test_consistent_number_of_rowers(self):
        matrix1 = torch.tensor([[1, 0, 0], [0, 1, 0]])
        matrix2 = torch.tensor([[1, 0], [0, 1]])
        with self.assertRaises(AssertionError):
            generate_valid_assignments([matrix1, matrix2])


class TestEvaluateSkillVariance(unittest.TestCase):

    def test_predefined_skills_and_assignments(self):
        assignments = torch.tensor([[[1, 0, 1], [0, 1, 1]]])  # 1 outing, 2 combinations, 3 rowers
        skills = torch.tensor([3, 5, 7])  # Skill levels
        variance_1 = torch.var(torch.tensor([3., 7]))
        variance_2 = torch.var(torch.tensor([5., 7]))
        expected_result = torch.tensor([variance_1, variance_2], dtype=torch.float16)
        result = evaluate_skill_variance(assignments, skills, dtype=torch.float16)
        self.assertTrue(torch.equal(result, expected_result))

    def test_multiple_boats(self):
        assignments = torch.tensor([[[1, 2, 1, 2], [2, 1, 1, 2], [1, 1, 2, 2]]])  # 1 outing, 3 combinations, 3 rowers
        skills = torch.tensor([3, 5, 7, 9])  # Skill levels
        variance_37 = torch.var(torch.tensor([3., 7]))
        variance_59 = torch.var(torch.tensor([5., 9]))
        variance_39 = torch.var(torch.tensor([3., 9]))
        variance_57 = torch.var(torch.tensor([5., 7]))
        variance_35 = torch.var(torch.tensor([3., 5]))
        variance_79 = torch.var(torch.tensor([7., 9]))
        expected_result = torch.tensor([
            variance_37 + variance_59,
            variance_39 + variance_57,
            variance_35 + variance_79
        ], dtype=torch.float16)
        result = evaluate_skill_variance(assignments, skills, dtype=torch.float16)
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, expected_result))

    def test_multiple_outings(self):
        assignments = torch.tensor([
            [[1, 0, 1], [0, 1, 1]],  # Outing 1
            [[1, 0, 1], [0, 1, 1]]   # Outing 2
        ])
        skills = torch.tensor([3, 5, 7])
        variance_1 = torch.var(torch.tensor([3., 7]))
        variance_2 = torch.var(torch.tensor([5., 7]))
        expected_result = torch.tensor([
            [2*variance_1, variance_2+variance_1],
            [variance_1+variance_2, 2*variance_2]
        ], dtype=torch.float16)
        result = evaluate_skill_variance(assignments, skills, dtype=torch.float16)
        self.assertTrue(torch.equal(result, expected_result))

    def test_edge_case_no_rowers_assigned(self):
        assignments = torch.tensor([[[0, 0, 0], [0, 0, 0]]])  # No rowers assigned
        skills = torch.tensor([3, 5, 7])
        result = evaluate_skill_variance(assignments, skills, dtype=torch.float16)
        # Expect zero variance as no rowers are assigned
        self.assertTrue(torch.all(result == 0))

    def test_edge_case_same_skill_level(self):
        assignments = torch.tensor([[[1, 0, 1], [0, 1, 1]]])
        skills = torch.tensor([5, 5, 5])  # All rowers have the same skill level
        result = evaluate_skill_variance(assignments, skills, dtype=torch.float16)
        # Expect zero variance as all rowers have the same skill level
        self.assertTrue(torch.all(result == 0))


class TestEvaluateNumPreferredOutings(unittest.TestCase):
    def test_predefined_assignments_and_preferences(self):
        assignments = torch.tensor([
            [[1,0,0], [0,1,0], [0,0,1]],  # Outing 1
            [[1,0,0], [0,1,0], [0,0,1]],  # Outing 2
        ])  
        preferred_outings = torch.tensor([0, 1, 2])
        expected_result = torch.tensor([
            [2, 1, 1],
            [1, 1, 0],
            [1, 0, 0]
        ])
        result = evaluate_num_preferred_outings(assignments, preferred_outings)
        self.assertTrue(torch.equal(result, expected_result))


class TestPermuteTopAssignments(unittest.TestCase):

    def test_permute_top_assignments(self):
        # Small, handcrafted example
        assignments_per_week = torch.tensor([
            [[1, 0, 0], [0, 1, 0]],  # Outing 1
            [[0, 1, 1], [1, 0, 1]]   # Outing 2
        ])
        total_score = torch.tensor([
            [0, 1],
            [3, 2]
        ])
        # this means that the best assignment has the 
        # index of [1, 0] in the score tensor
        # that translates to the assignment of
        #   outing 1 is [0, 1, 0] (the 1st combination of the 1st outing)
        #   outing 2 is [0, 1, 1] (the 0th combination of the 2nd outing)


        # The valid replacements are used for the permutation
        # to generate alternatives to a single outing at a time
        valid_assignments = [
            torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            torch.tensor([[0, 2, 2], [2, 0, 2], [2, 2, 0]])
        ]
        # Although the algorithm would never generate these assignments
        # because if there are two boats available they would need to be used
        # so this scenario is just for illustrative purposes.

        num_permutations = 3

        result = permute_top_assignments(
            valid_assignments, 
            assignments_per_week, 
            total_score, 
            num_permutations,
            randomize_permutations=False
        )
        num_outings, num_combinations, num_rowers = assignments_per_week.shape

        expected_shape = (num_outings, num_permutations, num_rowers)

        # The first permutation is the best assignment
        # The second - fourth are the valid replacements
        expected_result = torch.tensor([
            [[0, 1, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],  # Outing 1
            [[0, 1, 1], [0, 2, 2], [2, 0, 2], [2, 2, 0]],  # Outing 2
        ])
        self.assertTrue(torch.equal(result, expected_result))
        
        # Check that the first permutation is indeed the best assignment
        best_assignment = extract_best_assignment(assignments_per_week, total_score)
        self.assertTrue(torch.equal(result[:, :1, :], best_assignment))


# Running the test
if __name__ == '__main__':
    # run verbose tests
    unittest.main(verbosity=1)
    



