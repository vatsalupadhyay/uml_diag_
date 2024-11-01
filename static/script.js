// static/script.js
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('file-input');
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });
        const result = await response.json();

        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = "<h3>Validation Results:</h3>";
        if (result.error) {
            resultsDiv.innerHTML += `<p>Error: ${result.error}</p>`;
        } else {
            result.results.forEach((res, index) => {
                resultsDiv.innerHTML += `<p>${index + 1}. ${res}</p>`;
            });
        }
    } catch (error) {
        console.error("Error:", error);
    }
});
