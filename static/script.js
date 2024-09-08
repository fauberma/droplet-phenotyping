let selectedExperiment = null; // Store the selected experiment

// Function to fetch and display experiments in the sidebar
function fetchAndDisplayExperiments() {
    console.log("Fetching available experiments..."); // Debugging

    fetch('http://localhost:8080/get_experiments', {
        method: 'GET'
    })
    .then(response => {
        console.log("Response received:", response); // Debugging
        return response.json();
    })
    .then(data => {
        console.log("Data received:", data); // Debugging
        const experimentList = document.getElementById('experiment-list');
        experimentList.innerHTML = ''; // Clear the list first

        if (!Array.isArray(data) || data.length === 0) {
            console.log("No experiments found."); // Debugging
            return; // No experiments to display
        }

        // Loop through each experiment and add it to the list
        data.forEach(experiment => {
            const li = document.createElement('li');
            li.textContent = experiment;
            li.classList.add('experiment-item');

            // Add a click event listener to select the experiment
            li.addEventListener('click', function() {
                // Mark this as the selected experiment
                selectedExperiment = experiment;

                // Highlight the selected experiment
                document.querySelectorAll('.experiment-item').forEach(item => {
                    item.classList.remove('selected'); // Remove 'selected' from all
                });
                li.classList.add('selected'); // Highlight the clicked experiment
            });

            experimentList.appendChild(li);
        });
    })
    .catch(error => {
        console.error('Error fetching experiments:', error); // Log any errors
        alert('Failed to fetch experiments. Check console for details.');
    });
}


// Fetch models for Detect Outliers form and log responses for debugging
function fetchAndDisplayModelsForOutliers() {
    fetch('http://localhost:8080/get_models?model_type=outlier_detection', {method: 'GET'})
    .then(response => response.json())
    .then(data => {
        const modelDropdown = document.getElementById('model_name_do');
        modelDropdown.innerHTML = ''; // Clear any existing options

        // Populate the dropdown with models
        data.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelDropdown.appendChild(option);
        });
    })
    .catch(error => console.error('Error fetching models:', error));
}

// Fetch models for Detect Outliers form and log responses for debugging
function fetchAndDisplayModelsForCellCount() {
    fetch('http://localhost:8080/get_models?model_type=cell_count', {method: 'GET'})
    .then(response => response.json())
    .then(data => {
        const modelDropdown = document.getElementById('model_name_cc');
        modelDropdown.innerHTML = ''; // Clear any existing options

        // Populate the dropdown with models
        data.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelDropdown.appendChild(option);
        });
    })
    .catch(error => console.error('Error fetching models:', error));
}

// Helper function to display result in the 'result' div
function displayResult(result) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<h2>Response</h2><pre>${JSON.stringify(result, null, 2)}</pre>`;
}

// Ensure an experiment is selected before submitting the form
function ensureExperimentSelected() {
    if (!selectedExperiment) {
        alert('Please select an experiment first!');
        return false;
    }
    return true;
}

// Function to trigger Detect Droplets API
function triggerDetectDroplets(event) {
    event.preventDefault(); // Prevent the form from submitting in the default way
    if (!ensureExperimentSelected()) return;

    const mode = document.getElementById('mode_gdr').value; // Get the selected value from the dropdown

    fetch(`http://localhost:8080/detect_droplets?expID=${selectedExperiment}&mode=${mode}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(result => {
        displayResult(result);
    })
    .catch(error => console.error('Error:', error));
}

// Function to trigger Detect Outliers API
function triggerDetectOutliers(event) {
    event.preventDefault();
    if (!ensureExperimentSelected()) return;

    const modelName = document.getElementById('model_name_do').value;

    fetch(`http://localhost:8080/detect_outliers?expID=${selectedExperiment}&model_name=${modelName}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(result => {
        displayResult(result);
    })
    .catch(error => console.error('Error:', error));
}

// Function to trigger Cell Count API
function triggerCellCount(event) {
    event.preventDefault();
    if (!ensureExperimentSelected()) return;

    const modelName = document.getElementById('model_name_cc').value;

    fetch(`http://localhost:8080/cell_count?expID=${selectedExperiment}&model_name=${modelName}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(result => {
        displayResult(result);
    })
    .catch(error => console.error('Error:', error));
}

// Function to trigger Generate WP API
function triggerGenerateWP(event) {
    event.preventDefault();
    if (!ensureExperimentSelected()) return;

    const wpSize = document.getElementById('wp_size_wp').value;
    const excludeQuery = document.getElementById('exclude_query_wp').value;

    fetch(`http://localhost:8080/generate_wp?expID=${selectedExperiment}&wp_size=${wpSize}&exclude_query=${excludeQuery}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(result => {
        displayResult(result);
    })
    .catch(error => console.error('Error:', error));
}

// Add event listeners to the forms
function setupFormListeners() {
    document.getElementById('form_gdr').addEventListener('submit', triggerDetectDroplets);
    document.getElementById('form_do').addEventListener('submit', triggerDetectOutliers);
    document.getElementById('form_cc').addEventListener('submit', triggerCellCount);
    document.getElementById('form_wp').addEventListener('submit', triggerGenerateWP);
}

// Call the fetchAndDisplayExperiments function when the page loads and set up form listeners
document.addEventListener('DOMContentLoaded', () => {
    fetchAndDisplayExperiments();
    fetchAndDisplayModelsForOutliers();
    fetchAndDisplayModelsForCellCount();
    setupFormListeners();  // Set up form listeners after the page loads
});
