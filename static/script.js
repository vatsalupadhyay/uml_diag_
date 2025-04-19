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




// document.getElementById('upload-form').addEventListener('submit', async (e) => {
//     e.preventDefault();

//     const formData = new FormData();
//     const fileInput = document.getElementById('file-input');
//     formData.append('image', fileInput.files[0]); // Ensure the field name matches Flask's `request.files`

//     try {
//         // Make the POST request to the backend
//         const response = await fetch('/upload', {
//             method: 'POST',
//             body: formData,
//         });
//         const result = await response.json();

//         // Get the results div for displaying output
//         const resultsDiv = document.getElementById('results');
//         resultsDiv.innerHTML = "<h3>Validation Results:</h3>";

//         // Check if there's an error
//         if (result.error) {
//             resultsDiv.innerHTML += `<p style="color: red;">Error: ${result.error}</p>`;
//         } else {
//             // Display the validation results
//             const validationResults = result.results.validation || [];
//             if (validationResults.length > 0) {
//                 validationResults.forEach((res, index) => {
//                     resultsDiv.innerHTML += `<p>${index + 1}. ${res}</p>`;
//                 });
//             } else {
//                 resultsDiv.innerHTML += `<p>No validation issues detected.</p>`;
//             }

//             // Display the generated Java code if it exists
//             if (result.results.java_code) {
//                 const javaCodeSection = `
//                     <h3>Generated Java Code:</h3>
//                     <pre style="background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; overflow-x: auto;">
//                         ${result.results.java_code}
//                     </pre>
//                 `;
//                 resultsDiv.innerHTML += javaCodeSection;
//             } else {
//                 resultsDiv.innerHTML += `<p>No Java code was generated.</p>`;
//             }
//         }
//     } catch (error) {
//         console.error("Error:", error);

//         // Handle error display
//         const resultsDiv = document.getElementById('results');
//         resultsDiv.innerHTML = `<p style="color: red;">An error occurred while processing the file. Please try again.</p>`;
//     }
// });


