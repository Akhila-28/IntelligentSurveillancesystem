document.addEventListener('DOMContentLoaded', function() {
    const uploadButton = document.querySelector('#upload-button');
    const fileInput = document.getElementById('upload');
    const webcamButton = document.getElementById('webcam-button');

    // Handle file upload button click
    uploadButton.addEventListener('click', function() {
        fileInput.click();
    });

    // Handle file input change (when a file is selected)
    fileInput.addEventListener('change', function() {
        const form = document.getElementById('upload-form');
        const formData = new FormData(form);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('File uploaded successfully.');
                console.log('File saved at:', data.filename);

                fetch(`/detect_video/${data.filename}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Detection started in a new terminal.');
                        } else {
                            alert('Failed to start detection.');
                        }
                    });
            } else {
                alert('File upload failed.');
            }
        })
        .catch(error => console.error('Error:', error));
    });

    // Handle webcam button click
    webcamButton.addEventListener('click', function() {
        fetch('/start_crowd_webcam_detection')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Crowd detection webcam started.');
                } else {
                    alert('Failed to start crowd detection webcam.');
                }
            })
            .catch(error => console.error('Error:', error));
    });
});
