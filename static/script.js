let selectedExperiment = null; // Store the selected experiment
let allExperiments = []; // Store all experiments to allow filtering

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

// Function to fetch and display experiments in the sidebar
function fetchAndDisplayExperiments() {
    fetch('http://localhost:8080/get_experiments', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        allExperiments = data; // Store the fetched experiments
        displayFilteredExperiments(); // Display all experiments initially
    })
    .catch(error => {
        console.error('Error fetching experiments:', error);
        alert('Failed to fetch experiments. Check console for details.');
    });
}

// Function to display experiments based on the filter input
function displayFilteredExperiments() {
    const filterValue = document.getElementById('filter-input').value.toLowerCase();
    const experimentList = document.getElementById('experiment-list');
    experimentList.innerHTML = ''; // Clear the list first

    const filteredExperiments = allExperiments.filter(experiment =>
        experiment.toLowerCase().startsWith(filterValue)
    );

    if (filteredExperiments.length === 0) {
        const noResult = document.createElement('li');
        noResult.textContent = "No experiments found.";
        experimentList.appendChild(noResult);
    } else {
        filteredExperiments.forEach(experiment => {
            const li = document.createElement('li');
            li.textContent = experiment;
            li.classList.add('experiment-item');

            // Add a click event listener to select the experiment
            li.addEventListener('click', function() {
                selectedExperiment = experiment;

                document.querySelectorAll('.experiment-item').forEach(item => {
                    item.classList.remove('selected');
                });
                li.classList.add('selected');

                // Fetch and display the experiment data when an experiment is selected
                fetchExperimentData(selectedExperiment);
            });

            experimentList.appendChild(li);
        });
    }
}

function fetchExperimentData(expID) {
    fetch(`http://localhost:8080/get_experiment_data?expID=${expID}`, {
        method: 'GET'
    })
    .then(response => response.json()) // Assuming the response is in JSON format
    .then(data => {
        // Now you have the experiment data in 'data'
        // You can use it to populate a table or display the information as needed
        console.log(data); // For debugging purposes
        displayExperimentData(data); // Function to handle rendering the table
    })
    .catch(error => {
        console.error('Error fetching experiment data:', error);
    });
}

function displayExperimentData(data) {
    const tableContainer = document.getElementById('table-container');
    tableContainer.innerHTML = ''; // Clear previous data

    // Create the table
    const table = document.createElement('table');
    table.classList.add('experiment-data-table'); // Add class for styling

    // Create the header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    const headers = Object.keys(data[0]); // Assuming each row is an object
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create the body
    const tbody = document.createElement('tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(header => {
            const td = document.createElement('td');
            td.textContent = row[header]; // Add data to the cells
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    // Append the table to the container
    tableContainer.appendChild(table);
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

    const mode = document.getElementById('mode_dd').value; // Get the selected value from the dropdown

    fetch(`http://localhost:8080/detect_droplets?expID=${selectedExperiment}&mode=${mode}`, {
        method: 'POST'
    })
    .then(response => response.json())
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
    .catch(error => console.error('Error:', error));
}



// Add event listeners to the forms
function setupFormListeners() {
    document.getElementById('form_dd').addEventListener('submit', triggerDetectDroplets);
    document.getElementById('form_do').addEventListener('submit', triggerDetectOutliers);
    document.getElementById('form_cc').addEventListener('submit', triggerCellCount);
    document.getElementById('form_wp').addEventListener('submit', triggerGenerateWP);
}


// Add an event listener for filtering experiments
document.getElementById('filter-input').addEventListener('input', displayFilteredExperiments);


// Call the fetchAndDisplayExperiments function when the page loads and set up form listeners
document.addEventListener('DOMContentLoaded', () => {
    fetchAndDisplayExperiments();
    fetchAndDisplayModelsForOutliers();
    fetchAndDisplayModelsForCellCount();
    setupFormListeners();  // Set up form listeners after the page loads
});
