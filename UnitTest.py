import unittest
from unittest.mock import Mock, patch
from QueryProcessor.classes import QueryProcessor, Row, Rows, ExecutionResult
from QueryOptimizer.Nodes import ConditionLeaf, ConditionOperator
from QueryOptimizer.QueryTree import QueryTree
from StorageManager.classes import DataRetrieval, DataWrite, Condition
import json

class TestQueryProcessor(unittest.TestCase):
    def setUp(self):
        self.query_processor = QueryProcessor()
        
        # Setup test data based on the provided schema
        self.test_data = {
            'student': [
                Row({'student_id': 1, 'fullname': 'Hiyori Tomoe', 'gpa': 3.5}),
                Row({'student_id': 2, 'fullname': 'Natsume Sakasaki', 'gpa': 3.9}),
                Row({'student_id': 3, 'fullname': 'Daikichi', 'gpa': 2.0})
            ],
            'course': [
                Row({'course_id': 1, 'year': 2024, 'coursename': 'Kalkulus 1A', 'description': 'lorem ipsum'}),
                Row({'course_id': 2, 'year': 2020, 'coursename': 'Dasar Ilmu Sihir 2', 'description': 'dolor sit amet'}),
                Row({'course_id': 3, 'year': 2021, 'coursename': 'Sistem Basis Data', 'description': 'a' * 100})
            ],
            'attends': [
                Row({'student_id': 1, 'course_id': 3}),
                Row({'student_id': 2, 'course_id': 2}),
                Row({'student_id': 3, 'course_id': 1}),
                Row({'student_id': 2, 'course_id': 1}),
                Row({'student_id': 1, 'course_id': 1}),
                Row({'student_id': 1, 'course_id': 2}),
                Row({'student_id': 3, 'course_id': 3})
            ]
        }

    def test_parse_condition_with_schema_fields(self):
        """Test parsing conditions using actual schema fields"""
        test_cases = [
            {
                "condition": "student.gpa > 3.0",
                "expected": [{
                    "operand": "student.gpa",
                    "operator": ">",
                    "value": "3.0"
                }]
            },
            {
                "condition": "student.gpa > 3.0 AND student.student_id < 3",
                "expected": [
                    {
                        "operand": "student.gpa",
                        "operator": ">",
                        "value": "3.0"
                    },
                    {"logical_operator": "AND"},
                    {
                        "operand": "student.student_id",
                        "operator": "<",
                        "value": "3"
                    }
                ]
            },
            {
                "condition": "student.gpa > 3.0 OR student.student_id < 3",
                "expected": [
                    {
                        "operand": "student.gpa",
                        "operator": ">",
                        "value": "3.0"
                    },
                    {"logical_operator": "OR"},
                    {
                        "operand": "student.student_id",
                        "operator": "<",
                        "value": "3"
                    }
                ]
            },
            {
                "condition": "student.gpa > 3.0 AND student.student_id < 3 OR student.fullname = 'Hiyori Tomoe'",
                "expected": [
                    {
                        "operand": "student.gpa",
                        "operator": ">",
                        "value": "3.0"
                    },
                    {"logical_operator": "AND"},
                    {
                        "operand": "student.student_id",
                        "operator": "<",
                        "value": "3"
                    },
                    {"logical_operator": "OR"},
                    {
                        "operand": "student.fullname",
                        "operator": "=",
                        "value": "'Hiyori Tomoe'"
                    }
                ]
            }
        ]
        
        for case in test_cases:
            result = self.query_processor.parseCondition(case["condition"])
            self.assertEqual(result, case["expected"])

    def test_nested_loop_join_with_schema(self):
        """Test nested loop join using the student-course relationship"""
        # Get sample data
        attends = self.test_data['attends'] # First two students
        courses = self.test_data['course']    # First two courses
        
        result = self.query_processor.nested_loop_join(
            courses, attends, "attends.course_id", "course.course_id"
        )
        
        # Verify the join results
        self.assertIsInstance(result, list)
        for row in result:
            self.assertIsInstance(row, Row)
            self.assertTrue('course_id' in row.data)
            self.assertTrue('coursename' in row.data)
            self.assertTrue('description' in row.data)

    def test_hash_join_with_schema(self):
        """Test hash join using student-attends relationship"""
        students = self.test_data['student']
        attends = self.test_data['attends']
        
        result = self.query_processor.hash_join(
            students, attends, "student.student_id", "attends.student_id"
        )
        
        # Verify join results
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        for row in result:
            self.assertIsInstance(row, Row)
            self.assertTrue('student_id' in row.data)
            self.assertTrue('course_id' in row.data)
            self.assertTrue('fullname' in row.data)

    def test_cartesian_with_schema(self):
        """Test cartesian product using actual schema tables"""
        students = self.test_data['student']  # First two students
        courses = self.test_data['course']   # First two courses
        
        result = self.query_processor.cartesian(students, courses)
        
        self.assertEqual(len(result), 9)
        for row in result:
            self.assertTrue('student_id' in row.data)
            self.assertTrue('course_id' in row.data)
            self.assertTrue('fullname' in row.data)
            self.assertTrue('coursename' in row.data)

    def test_convert_value(self):
        """Test value conversion with schema-specific data types"""
        test_cases = [
            ("3.5", 3.5),          # GPA (float)
            ("2024", 2024.0),      # Year (int)
            ("Hiyori Tomoe", "Hiyori Tomoe"),  # Fullname (varchar)
            ("1", 1.0),            # ID (int)
        ]
        
        for input_val, expected in test_cases:
            result = self.query_processor.convert_value(input_val)
            self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()