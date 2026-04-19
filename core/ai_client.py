import base64
import logging
import time
import requests
from PIL.Image import Image
from typing import Optional
from io import BytesIO
import json

logger = logging.getLogger(__name__)


class GeminiVisionClient:
    """Client for Gemini Vision API integration using REST API."""

    # Gemini API endpoint
    API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str = "gemini-flash-latest", timeout_s: float = 60.0):
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model: Model name to use (default: gemini-flash-latest)
            timeout_s: HTTP read timeout per request (default 60s)
        """
        self.api_key = api_key
        self.model_name = model
        self.timeout_s = float(timeout_s)
        self.api_url = f"{self.API_BASE_URL}/{model}:generateContent"
        logger.info(f"Initialized Gemini client with model: {model} (timeout={self.timeout_s}s)")

    def analyze(self, image: Image, prompt: str, max_retries: int = 3) -> str:
        """Analyze an image with a prompt using Gemini Vision.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the analysis
            max_retries: Maximum number of retries on failure

        Returns:
            Response text from the model
        """
        image_base64 = self._image_to_base64(image)

        for attempt in range(max(1, max_retries)):
            try:
                logger.debug(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})")

                # Prepare request
                headers = {
                    "Content-Type": "application/json",
                    "X-goog-api-key": self.api_key,
                }

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image_base64,
                                    }
                                },
                                {
                                    "text": prompt
                                },
                            ]
                        }
                    ]
                }

                # Send request
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )

                response.raise_for_status()

                # Extract text from response
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            text = parts[0]["text"]
                            logger.debug(f"Got response from Gemini: {text[:100]}...")
                            return text

                logger.error(f"Unexpected API response format: {result}")
                raise ValueError("Invalid response format from API")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited
                    logger.warning(f"Rate limited by Gemini API: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    logger.error(f"HTTP error from Gemini API: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise

            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed to analyze image after {max_retries} attempts")

    def analyze_text(self, prompt: str, max_retries: int = 3) -> str:
        """Send text-only prompt to Gemini (no image).

        Args:
            prompt: Text prompt for the analysis
            max_retries: Maximum number of retries on failure

        Returns:
            Response text from the model
        """
        for attempt in range(max(1, max_retries)):
            try:
                logger.debug(f"Calling Gemini API with text-only prompt (attempt {attempt + 1}/{max_retries})")

                # Prepare request - text only, no image
                headers = {
                    "Content-Type": "application/json",
                    "X-goog-api-key": self.api_key,
                }

                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }

                response = requests.post(self.api_url, json=payload, headers=headers, timeout=self.timeout_s)
                response.raise_for_status()

                # Extract text from response
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if len(parts) > 0 and "text" in parts[0]:
                            text = parts[0]["text"]
                            logger.debug(f"Got response from Gemini: {text[:100]}...")
                            return text

                logger.error(f"Unexpected API response format: {result}")
                raise ValueError("Invalid response format from API")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    # Rate limited
                    logger.warning(f"Rate limited by Gemini API: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    logger.error(f"HTTP error from Gemini API: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise

            except Exception as e:
                logger.error(f"Error calling Gemini API: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed to analyze text after {max_retries} attempts")

    @staticmethod
    def _image_to_base64(image: Image) -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded image string
        """
        # Convert image to JPEG
        buffered = BytesIO()
        # Ensure image is in RGB mode for JPEG
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()

        # Encode to base64
        return base64.standard_b64encode(img_bytes).decode("utf-8")


class LocalVisionClient:
    """Client for local vision models (OpenAI-compatible API)."""

    def __init__(self, api_base: str = "http://192.168.1.156:1234", model: str = "qwen/qwen3-vl-8b",
                 image_format: str = "jpeg", image_quality: int = 85):
        """Initialize local vision client.

        Args:
            api_base: Base URL for the local API (default: LM Studio)
            model: Model name to use (default: qwen/qwen3-vl-8b)
            image_format: Image format for encoding ("jpeg" or "webp")
            image_quality: Image quality (0-100)
        """
        self.api_base = api_base.rstrip("/")
        self.model_name = model
        self.api_url = f"{self.api_base}/v1/chat/completions"
        self.image_format = image_format.upper()  # "JPEG" or "WEBP"
        self.image_quality = image_quality
        logger.info(f"Initialized LocalVisionClient with model: {model} at {self.api_base} (format={self.image_format}, quality={image_quality})")

    def analyze(self, image: Image, prompt: str, max_retries: int = 1) -> str:
        """Analyze an image with a prompt using local vision model.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the analysis
            max_retries: Maximum number of retries on failure

        Returns:
            Response text from the model
        """
        image_base64 = self._encode_image(image)

        for attempt in range(max(1, max_retries)):
            try:
                logger.debug(f"Calling local model API (attempt {attempt + 1}/{max_retries})")

                # Prepare request in OpenAI-compatible format
                headers = {
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                }

                # Send request
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=120,  # Longer timeout for local inference
                )

                response.raise_for_status()

                # Extract text from response
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        text = choice["message"]["content"]
                        logger.debug(f"Got response from local model: {text[:100]}...")
                        return text

                logger.error(f"Unexpected API response format: {result}")
                raise ValueError("Invalid response format from API")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout calling local model API")
                if attempt < max_retries - 1:
                    wait_time = 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error to local model API: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                logger.error(f"Error calling local model API: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed to analyze image after {max_retries} attempts")

    def analyze_text(self, prompt: str, max_retries: int = 1) -> str:
        """Send text-only prompt to local model (no image).

        Args:
            prompt: Text prompt for the analysis
            max_retries: Maximum number of retries on failure

        Returns:
            Response text from the model
        """
        for attempt in range(max(1, max_retries)):
            try:
                logger.debug(f"Calling local model API with text-only prompt (attempt {attempt + 1}/{max_retries})")

                headers = {
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3,
                }

                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        text = choice["message"]["content"]
                        logger.debug(f"Got response from local model: {text[:100]}...")
                        return text

                logger.error(f"Unexpected API response format: {result}")
                raise ValueError("Invalid response format from API")

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout calling local model API")
                if attempt < max_retries - 1:
                    wait_time = 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

            except Exception as e:
                logger.error(f"Error calling local model API: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed to analyze text after {max_retries} attempts")

    def _encode_image(self, image: Image) -> str:
        """Convert PIL Image to base64 string with configured format.

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded image string
        """
        buffered = BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format=self.image_format, quality=self.image_quality)
        img_bytes = buffered.getvalue()
        return base64.standard_b64encode(img_bytes).decode("utf-8")

    @staticmethod
    def _image_to_base64(image: Image) -> str:
        """Convert PIL Image to base64 string (JPEG, backward compatibility).

        Args:
            image: PIL Image object

        Returns:
            Base64 encoded image string
        """
        buffered = BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        return base64.standard_b64encode(img_bytes).decode("utf-8")
