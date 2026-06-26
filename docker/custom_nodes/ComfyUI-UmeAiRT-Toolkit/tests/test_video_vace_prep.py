import pytest
from unittest.mock import patch, MagicMock

# Attempt to import torch and UmeAiRT_VideoVacePrep. If it fails, skip all tests.
try:
    import torch
    from modules.video_vace_prep import UmeAiRT_VideoVacePrep
    from modules.common import UmeVaceFrames
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not COMFY_AVAILABLE, reason="ComfyUI environment not available")


@pytest.fixture
def vace_prep_node():
    return UmeAiRT_VideoVacePrep()


def test_vace_prep_init(vace_prep_node):
    """Test node initialization and basic properties."""
    assert vace_prep_node.CATEGORY == "UmeAiRT/Block/Inputs"
    assert vace_prep_node.RETURN_TYPES == ("UME_VACE_FRAMES",)
    
    inputs = UmeAiRT_VideoVacePrep.INPUT_TYPES()
    assert "required" in inputs
    assert "start_image" in inputs["required"]
    assert "optional" in inputs
    assert "end_image" in inputs["optional"]


def test_vace_prep_process_start_only(vace_prep_node):
    """Test processing with only a start image."""
    mock_start = torch.ones((1, 64, 64, 3))
    
    result = vace_prep_node.process(start_image=mock_start)
    
    assert isinstance(result, tuple)
    assert len(result) == 1
    
    vace_frames = result[0]
    assert isinstance(vace_frames, UmeVaceFrames)
    assert torch.equal(vace_frames.start_image, mock_start)
    assert vace_frames.end_image is None


def test_vace_prep_process_start_and_end(vace_prep_node):
    """Test processing with both start and end images."""
    mock_start = torch.ones((1, 64, 64, 3))
    mock_end = torch.zeros((1, 64, 64, 3))
    
    result = vace_prep_node.process(start_image=mock_start, end_image=mock_end)
    
    vace_frames = result[0]
    assert torch.equal(vace_frames.start_image, mock_start)
    assert torch.equal(vace_frames.end_image, mock_end)


def test_vace_prep_invalid_input(vace_prep_node):
    """Test error handling when required inputs are missing or invalid."""
    # Start image is required, but let's pass None explicitly (should ideally fail before this, but just in case)
    with pytest.raises(Exception):
        vace_prep_node.process(start_image=None)
