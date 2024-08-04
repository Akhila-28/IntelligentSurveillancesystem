// Handle the click event for the upload button
document.getElementById('fall-upload-button').addEventListener('click', function() {
    document.getElementById('fall-upload').click();
});

// Handle the file upload
document.getElementById('fall-upload').addEventListener('change', function() {
    var form = document.getElementById('fall-upload-form');
    var formData = new FormData(form);

    fetch('/upload_fall_video', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
      .then(data => {
        if (data.success) {
            alert('Video uploaded successfully!');
            startFallDetection(data.filename);
        } else {
            alert('Error uploading video: ' + data.message);
        }
    }).catch(error => {
        console.error('Error:', error);
    });
});

function startFallDetection(filename) {
    fetch(`/detect_fall_video/${filename}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Fall detection started.');
        } else {
            alert('Error starting fall detection: ' + data.message);
        }
    }).catch(error => {
        console.error('Error:', error);
    });
}

// Handle the click event for starting fall webcam detection
document.getElementById('fall-webcam-button').addEventListener('click', function() {
    fetch('/start_fall_webcam_detection')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Fall detection webcam started.');
        } else {
            alert('Error starting fall detection webcam: ' + data.message);
        }
    }).catch(error => {
        console.error('Error:', error);
    });
});

// Function to start video detection
function startVideoDetection(filename) {
    fetch(`/detect_video/${filename}`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Video detection started.');
        } else {
            alert('Error starting video detection: ' + data.message);
        }
    }).catch(error => {
        console.error('Error:', error);
    });
}
