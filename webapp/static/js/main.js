document.addEventListener("DOMContentLoaded", function () {
  // Elements
  const uploadArea = document.getElementById("upload-area");
  const uploadPrompt = document.getElementById("upload-prompt");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const fileInput = document.getElementById("file-input");
  const uploadButton = document.getElementById("upload-button");
  const cameraButton = document.getElementById("camera-button");
  const resetButton = document.getElementById("reset-button");
  const recognizeButton = document.getElementById("recognize-button");
  const resultsEmpty = document.getElementById("results-empty");
  const resultsLoading = document.getElementById("results-loading");
  const resultsError = document.getElementById("results-error");
  const resultsContent = document.getElementById("results-content");
  const errorMessage = document.getElementById("error-message");
  const recognizedText = document.getElementById("recognized-text");
  const confidenceBar = document.getElementById("confidence-bar");
  const confidenceValue = document.getElementById("confidence-value");
  const confidenceLabel = document.getElementById("confidence-label");
  const confidenceMethod = document.getElementById("confidence-method");
  const confidenceHeatmap = document.getElementById("confidence-heatmap");
  const methodDetails = document.getElementById("method-details");
  const sampleImages = document.querySelectorAll(".sample-image");

  // Camera elements
  const cameraModal = new bootstrap.Modal(
    document.getElementById("camera-modal"),
  );
  const cameraView = document.getElementById("camera-view");
  const cameraCanvas = document.getElementById("camera-canvas");
  const captureButton = document.getElementById("capture-button");

  let stream = null;

  // Event listeners for drag and drop
  uploadArea.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", function () {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadArea.classList.remove("dragover");

    if (e.dataTransfer.files.length) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  // Click to upload
  uploadButton.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    if (fileInput.files.length) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  // Camera capture
  cameraButton.addEventListener("click", function () {
    openCamera();
  });

  captureButton.addEventListener("click", function () {
    captureImage();
  });

  // Reset button
  resetButton.addEventListener("click", function () {
    resetUpload();
  });

  // Recognize button
  recognizeButton.addEventListener("click", function () {
    recognizeImage();
  });

  // Sample images
  sampleImages.forEach((image) => {
    image.addEventListener("click", function () {
      processSampleImage(this.dataset.filename);
    });
  });

  // Handle file selection
  function handleFileSelect(file) {
    if (!file.type.match("image.*")) {
      showError("Please select an image file");
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      showPreview();
    };
    reader.readAsDataURL(file);
  }

  // Show preview
  function showPreview() {
    uploadPrompt.classList.add("d-none");
    previewContainer.classList.remove("d-none");
    animateElement(previewContainer);
  }

  // Reset upload
  function resetUpload() {
    previewContainer.classList.add("d-none");
    uploadPrompt.classList.remove("d-none");
    fileInput.value = "";
    animateElement(uploadPrompt);
  }

  // Camera functions
  function openCamera() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then(function (mediaStream) {
          stream = mediaStream;
          cameraView.srcObject = mediaStream;
          cameraModal.show();
        })
        .catch(function (error) {
          console.error("Camera error:", error);
          showError("Could not access camera");
        });
    } else {
      showError("Camera not supported in your browser");
    }
  }

  function captureImage() {
    const context = cameraCanvas.getContext("2d");

    // Set canvas dimensions to match video
    cameraCanvas.width = cameraView.videoWidth;
    cameraCanvas.height = cameraView.videoHeight;

    // Draw video frame to canvas
    context.drawImage(
      cameraView,
      0,
      0,
      cameraCanvas.width,
      cameraCanvas.height,
    );

    // Get image data
    const imageData = cameraCanvas.toDataURL("image/png");

    // Display in preview
    previewImage.src = imageData;
    showPreview();

    // Close camera
    closeCamera();
  }

  function closeCamera() {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    cameraModal.hide();
  }

  // Recognize image
  function recognizeImage() {
    showLoading();

    // Get the selected confidence method
    const method = confidenceMethod.value;

    // Create form data
    const formData = new FormData();

    // Check if we're using a file or canvas data
    if (previewImage.src.startsWith("data:image")) {
      // It's a data URL, extract the base64 data
      formData.append("image_data", previewImage.src);
    } else {
      // Convert the image to a blob
      fetch(previewImage.src)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "image.png", { type: "image/png" });
          formData.append("image", file);
          sendRecognizeRequest(formData, method);
        })
        .catch((error) => {
          console.error("Error fetching image:", error);
          showError("Error processing image");
        });
      return;
    }

    // Add confidence method
    formData.append("confidence_method", method);

    // Send request
    sendRecognizeRequest(formData, method);
  }

  function sendRecognizeRequest(formData, method) {
    fetch("/api/recognize", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        if (data.error) {
          showError(data.error);
        } else {
          displayResults(data);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        showError("Failed to process image");
      });
  }

  // Process sample image
  function processSampleImage(filename) {
    previewImage.src = `/static/samples/${filename}`;
    showPreview();

    // Now get the recognition results
    showLoading();

    fetch(`/api/sample/${filename}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        if (data.error) {
          showError(data.error);
        } else {
          displayResults(data);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        showError("Failed to process sample image");
      });
  }

  // Display states
  function showLoading() {
    resultsEmpty.classList.add("d-none");
    resultsError.classList.add("d-none");
    resultsContent.classList.add("d-none");
    resultsLoading.classList.remove("d-none");
  }

  function showError(message) {
    resultsEmpty.classList.add("d-none");
    resultsLoading.classList.add("d-none");
    resultsContent.classList.add("d-none");

    errorMessage.textContent = message;
    resultsError.classList.remove("d-none");
    animateElement(resultsError);
  }

  function displayResults(data) {
    resultsEmpty.classList.add("d-none");
    resultsLoading.classList.add("d-none");
    resultsError.classList.add("d-none");

    // Set recognized text
    recognizedText.textContent = data.text || "No text recognized";

    // Set confidence
    const confidence = data.overall_confidence;
    confidenceBar.style.width = `${confidence * 100}%`;
    confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;

    // Color the confidence bar based on value
    setConfidenceColors(confidenceBar, confidence);

    // Set confidence label
    confidenceLabel.textContent = getConfidenceLabel(confidence);

    // Set confidence heatmap
    if (data.confidence_heatmap) {
      confidenceHeatmap.src = data.confidence_heatmap;
    } else {
      confidenceHeatmap.src = "";
    }

    // Method-specific details
    displayMethodDetails(data);

    // Show content
    resultsContent.classList.remove("d-none");
    animateElement(resultsContent);
  }

  function displayMethodDetails(data) {
    let detailsHTML = "";

    switch (data.method) {
      case "temperature":
        detailsHTML = `
                    <div class="alert alert-info">
                        <h6 class="alert-heading"><i class="fas fa-thermometer-half me-2"></i> Temperature Scaling</h6>
                        <p class="mb-0">A calibration technique that uses a single scalar parameter (temperature) to make model predictions better calibrated.</p>
                        <hr>
                        <p class="mb-0"><strong>Temperature Value:</strong> ${data.temperature?.toFixed(2) || "N/A"}</p>
                    </div>
                `;
        break;

      case "step_dependent":
        // Format temperatures for display
        let tempHTML = "";
        if (data.temperatures && data.temperatures.length > 0) {
          tempHTML =
            '<div class="mt-2"><strong>Position-specific Temperatures:</strong><br>';
          data.temperatures.forEach((temp, idx) => {
            if (idx < 10) {
              tempHTML += `<span class="badge bg-info me-1 mb-1">Pos ${idx}: ${temp.toFixed(2)}</span>`;
            }
          });
          tempHTML += "</div>";
        }

        detailsHTML = `
                    <div class="alert alert-primary">
                        <h6 class="alert-heading"><i class="fas fa-layer-group me-2"></i> Step-Dependent Temperature Scaling</h6>
                        <p class="mb-0">An advanced calibration technique that applies different temperatures to different positions in the sequence.</p>
                        <hr>
                        ${tempHTML}
                    </div>
                `;
        break;

      case "mc_dropout":
        detailsHTML = `
                    <div class="alert alert-warning">
                        <h6 class="alert-heading"><i class="fas fa-random me-2"></i> Monte Carlo Dropout</h6>
                        <p class="mb-0">Uses multiple forward passes with dropout to estimate model uncertainty.</p>
                        <hr>
                        <p class="mb-0"><strong>Epistemic Uncertainty:</strong> ${data.epistemic_uncertainty?.toFixed(4) || "N/A"} (model uncertainty)</p>
                        <p class="mb-0"><strong>Aleatoric Uncertainty:</strong> ${data.aleatoric_uncertainty?.toFixed(4) || "N/A"} (data uncertainty)</p>
                    </div>
                `;
        break;

      case "uncalibrated":
        detailsHTML = `
                    <div class="alert alert-secondary">
                        <h6 class="alert-heading"><i class="fas fa-exclamation-triangle me-2"></i> Uncalibrated Confidence</h6>
                        <p class="mb-0">Raw softmax probabilities from the model without any calibration. These tend to be overconfident.</p>
                    </div>
                `;
        break;

      default:
        detailsHTML = `
                    <div class="alert alert-secondary">
                        <h6 class="alert-heading">Confidence Method: ${data.method || "Unknown"}</h6>
                        <p class="mb-0">No additional details available for this method.</p>
                    </div>
                `;
    }

    methodDetails.innerHTML = detailsHTML;
  }

  // Utility functions
  function setConfidenceColors(element, confidence) {
    // Remove existing classes
    element.classList.remove(
      "bg-danger",
      "bg-warning",
      "bg-info",
      "bg-success",
    );

    // Add appropriate class based on confidence value
    if (confidence < 0.5) {
      element.classList.add("bg-danger");
    } else if (confidence < 0.7) {
      element.classList.add("bg-warning");
    } else if (confidence < 0.9) {
      element.classList.add("bg-info");
    } else {
      element.classList.add("bg-success");
    }
  }

  function getConfidenceLabel(confidence) {
    if (confidence < 0.3) return "Very Low Confidence";
    if (confidence < 0.5) return "Low Confidence";
    if (confidence < 0.7) return "Moderate Confidence";
    if (confidence < 0.9) return "High Confidence";
    return "Very High Confidence";
  }

  function animateElement(element) {
    anime({
      targets: element,
      opacity: [0, 1],
      translateY: [20, 0],
      easing: "easeOutQuad",
      duration: 500,
    });
  }

  // Handle modal close
  document
    .getElementById("camera-modal")
    .addEventListener("hidden.bs.modal", function () {
      closeCamera();
    });
});
