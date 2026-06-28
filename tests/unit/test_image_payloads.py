import pytest

from src.image_payloads import ImagePayloadError, image_data_url


def test_image_data_url_encodes_local_image(tmp_path):
    image_path = tmp_path / "uav.png"
    image_path.write_bytes(b"image")

    assert image_data_url(image_path) == "data:image/png;base64,aW1hZ2U="


def test_image_data_url_fails_for_unknown_file_type(tmp_path):
    image_path = tmp_path / "uav"
    image_path.write_bytes(b"image")

    with pytest.raises(ImagePayloadError, match="unsupported image type"):
        image_data_url(image_path)
