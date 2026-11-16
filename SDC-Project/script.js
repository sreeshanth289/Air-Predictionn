
const form = document.getElementById("predictionForm");

const predictBtn = document.getElementById("predictBtn");

const prediction = document.getElementById("prediction");

const message = document.getElementById("message");

form.addEventListener("submit", async function(event) {

    event.preventDefault();

    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    prediction.textContent = "--";
    message.textContent = "Connecting to ML model...";

    // Collect input values
    const formData = new FormData(form);

    const features = {};

    for (const [key, value] of formData.entries()) {
        features[key] = Number(value);
    }

    // Prepare API data
    const data = {
        features: features
    };

    try {

        // Send request to FastAPI
        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );

        if (!response.ok) {
            throw new Error("API request failed: " + response.status);
        }

        // Get prediction
        const result = await response.json();

        // Display prediction
        prediction.textContent = result.predicted_CO.toFixed(4);

        message.textContent =
            "Prediction generated successfully by your ML model.";

    }

    catch(error) {

        console.error("Prediction Error:", error);

        prediction.textContent = "--";

        message.textContent =
            "Unable to connect to the ML model. Check whether FastAPI is running.";

    }

    finally {

        predictBtn.disabled = false;

        predictBtn.textContent = "Predict CO Concentration →";

    }

});
