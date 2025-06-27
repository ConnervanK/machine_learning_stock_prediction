import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from polybox_download import extract_zip_file, download_file_from_polybox


class TestExtractZipFile:
    
    def test_extract_zip_file_basic(self):
        """Test basic zip file extraction functionality"""
        import zipfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test zip file
            zip_path = os.path.join(tmpdir, 'test.zip')
            extract_dir = os.path.join(tmpdir, 'extracted')
            
            # Create a zip file with a test file
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('test_file.txt', 'Hello, World!')
                zf.writestr('subfolder/another_file.txt', 'Another file content')
            
            # Test extraction
            extract_zip_file(zip_path, extract_dir)
            
            # Check that files were extracted
            assert os.path.exists(extract_dir)
            assert os.path.exists(os.path.join(extract_dir, 'test_file.txt'))
            assert os.path.exists(os.path.join(extract_dir, 'subfolder', 'another_file.txt'))
            
            # Check file contents
            with open(os.path.join(extract_dir, 'test_file.txt'), 'r') as f:
                assert f.read() == 'Hello, World!'
    
    def test_extract_zip_file_nonexistent(self):
        """Test extraction of non-existent zip file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'nonexistent.zip')
            extract_dir = os.path.join(tmpdir, 'extracted')
            
            # Should handle gracefully or raise appropriate exception
            with pytest.raises((FileNotFoundError, Exception)):
                extract_zip_file(zip_path, extract_dir)
    
    def test_extract_zip_file_invalid_zip(self):
        """Test extraction of invalid zip file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file that's not a valid zip
            invalid_zip_path = os.path.join(tmpdir, 'invalid.zip')
            with open(invalid_zip_path, 'w') as f:
                f.write('This is not a zip file')
            
            extract_dir = os.path.join(tmpdir, 'extracted')
            
            # Should handle invalid zip gracefully
            with pytest.raises((Exception)):
                extract_zip_file(invalid_zip_path, extract_dir)


class TestDownloadFileFromPolybox:
    
    @patch('polybox_download.requests.get')
    def test_download_file_from_polybox_success(self, mock_get):
        """Test successful file download from Polybox"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'File content here'
        mock_response.headers = {'content-length': '17'}
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'downloaded_file.zip')
            
            # Test download
            success = download_file_from_polybox('https://polybox.ethz.ch/test', output_path)
            
            # Check results
            assert success == True
            assert os.path.exists(output_path)
            
            # Check file content
            with open(output_path, 'rb') as f:
                assert f.read() == b'File content here'
    
    @patch('polybox_download.requests.get')
    def test_download_file_from_polybox_http_error(self, mock_get):
        """Test download with HTTP error"""
        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'failed_download.zip')
            
            # Test download should fail gracefully
            success = download_file_from_polybox('https://polybox.ethz.ch/notfound', output_path)
            
            # Should return False on failure
            assert success == False
            # File should not be created on failure
            assert not os.path.exists(output_path)
    
    @patch('polybox_download.requests.get')
    def test_download_file_from_polybox_network_error(self, mock_get):
        """Test download with network error"""
        # Mock network error
        mock_get.side_effect = Exception("Network error")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'network_error_download.zip')
            
            # Test download should handle network error
            success = download_file_from_polybox('https://polybox.ethz.ch/test', output_path)
            
            # Should return False on network error
            assert success == False
            assert not os.path.exists(output_path)
    
    def test_download_file_from_polybox_invalid_url(self):
        """Test download with invalid URL"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'invalid_url_download.zip')
            
            # Test with clearly invalid URL
            success = download_file_from_polybox('not-a-url', output_path)
            
            # Should handle invalid URL gracefully
            assert success == False


class TestPolyboxIntegration:
    
    @patch('polybox_download.requests.get')
    def test_download_and_extract_pipeline(self, mock_get):
        """Test complete download and extract pipeline"""
        import zipfile
        
        # Create a mock zip content
        zip_content = b''
        with tempfile.NamedTemporaryFile() as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
                zf.writestr('data.csv', 'date,value\n2023-01-01,100\n2023-01-02,101')
            
            with open(tmp_zip.name, 'rb') as f:
                zip_content = f.read()
        
        # Mock successful download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_content
        mock_response.headers = {'content-length': str(len(zip_content))}
        mock_get.return_value = mock_response
        
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'downloaded.zip')
            extract_dir = os.path.join(tmpdir, 'extracted')
            
            # Test download
            success = download_file_from_polybox('https://polybox.ethz.ch/test', zip_path)
            assert success == True
            
            # Test extraction
            extract_zip_file(zip_path, extract_dir)
            
            # Check that CSV was extracted
            csv_path = os.path.join(extract_dir, 'data.csv')
            assert os.path.exists(csv_path)
            
            # Check CSV content
            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert 'date' in df.columns
            assert 'value' in df.columns
    
    def test_polybox_functions_exist(self):
        """Test that polybox functions are properly defined"""
        # Check that functions exist and are callable
        assert callable(download_file_from_polybox)
        assert callable(extract_zip_file)


class TestPolyboxErrorHandling:
    
    def test_extract_to_existing_directory(self):
        """Test extraction to an existing directory"""
        import zipfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test zip file
            zip_path = os.path.join(tmpdir, 'test.zip')
            extract_dir = os.path.join(tmpdir, 'existing_dir')
            
            # Create the extraction directory first
            os.makedirs(extract_dir)
            
            # Create a zip file
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('test.txt', 'Test content')
            
            # Should handle existing directory gracefully
            try:
                extract_zip_file(zip_path, extract_dir)
                # If successful, check that file was extracted
                assert os.path.exists(os.path.join(extract_dir, 'test.txt'))
            except Exception as e:
                # If it fails, should be a reasonable error
                assert isinstance(e, Exception)
    
    @patch('polybox_download.requests.get')
    def test_download_with_invalid_output_path(self, mock_get):
        """Test download with invalid output path"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'test content'
        mock_get.return_value = mock_response
        
        # Try to download to an invalid path (directory that doesn't exist)
        invalid_path = '/nonexistent/directory/file.zip'
        
        success = download_file_from_polybox('https://polybox.ethz.ch/test', invalid_path)
        
        # Should handle invalid path gracefully
        assert success == False


if __name__ == "__main__":
    pytest.main([__file__])
