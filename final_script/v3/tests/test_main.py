import pytest
import os
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

# Import the modules we want to test
from final_script.v3.modules.log_config import track_time

class TestLogConfig:
    @patch('final_script.v3.modules.log_config.logger')
    def test_track_time_decorator(self, mock_logger):
        """Test that track_time decorator logs execution time"""
        
        # Create a test function to decorate
        @track_time
        def test_function():
            return "test result"
        
        # Call the decorated function
        result, elapsed_time = test_function()
        
        # Assertions
        assert result == "test result"
        assert isinstance(elapsed_time, float)
        mock_logger.info.assert_called_once()
        
        # Check if the log message contains the function name and time
        log_message = mock_logger.info.call_args[0][0]
        assert "test_function" in log_message
        assert "seconds" in log_message

    @patch('final_script.v3.modules.log_config.logger')
    def test_track_time_with_args(self, mock_logger):
        """Test track_time decorator with function arguments"""
        
        @track_time
        def test_function_with_args(x, y, name="test"):
            return f"{name}: {x + y}"
        
        # Call with different types of arguments
        result, elapsed_time = test_function_with_args(1, 2, name="sum")
        
        # Assertions
        assert result == "sum: 3"
        assert isinstance(elapsed_time, float)
        mock_logger.info.assert_called_once()

    @patch('final_script.v3.modules.log_config.logger')
    def test_track_time_with_error(self, mock_logger):
        """Test track_time decorator when function raises an error"""
        
        @track_time
        def error_function():
            raise ValueError("Test error")
        
        # Test that the error is propagated
        with pytest.raises(ValueError) as exc_info:
            error_function()
        
        assert str(exc_info.value) == "Test error"
        # Verify that no time was logged (since function errored)
        mock_logger.info.assert_not_called()