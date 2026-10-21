
const form = document.getElementById("prediction-form");
const result = document.getElementById("result");
const message = document.getElementById("message");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    result.hidden = true;
    message.textContent = "Predicting...";

    const input = {
        consumption_kwh: Number(
            document.getElementById("consumption").value
        ),
        hour: Number(document.getElementById("hour").value),
        day_of_week: Number(document.getElementById("day").value),
        tariff_per_kwh: Number(document.getElementById("tariff").value)
    };

    try {
        const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(input)
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        document.getElementById("usage").textContent =
            data.predicted_consumption_kwh + " kWh";

        document.getElementById("cost").textContent =
            "₹" + data.estimated_cost_inr.toFixed(2);

        result.hidden = false;
        message.textContent = "";
    } catch (error) {
        message.textContent =
            "Could not get prediction. Check the API is running.";
    }
});
