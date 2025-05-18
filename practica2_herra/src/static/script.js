document.getElementById("train-form").addEventListener("submit", async function (e) {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target).entries());
    data.n_estimators = parseInt(data.n_estimators);
    data.max_depth = parseInt(data.max_depth);

    document.getElementById("train-result").innerText = "";
    const res = await fetch("/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });
    console.log(JSON.stringify(data))

    const json = await res.json();
    document.getElementById("train-result").innerText = json.message || json.error;
});

document.getElementById("predict-form").addEventListener("submit", async function(e) {
    e.preventDefault();
    const data = Object.fromEntries(new FormData(e.target).entries());

    document.getElementById("prediction-result").innerText = ""
    for (let key in data) {
        data[key] = parseFloat(data[key]);
    }

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const json = await res.json();
    document.getElementById("prediction-result").innerText =
        json.prediction !== undefined ? "Se predice que puede pasar un fallo cardiaco? " + json.prediction : json.error;
});

document.getElementById("test").addEventListener("click", async function(e) {
    e.preventDefault();

    document.getElementById("testing-data").innerText = "";
    document.getElementById("prediction-result2").innerText = "";
    document.getElementById("prediction-result3").innerText = "";
    const res = await fetch("/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    });

    const json = await res.json();
    document.getElementById("testing-data").innerText =
        json.prediction !== undefined ? json.data : json.error;
    document.getElementById("prediction-result2").innerText =
        json.prediction !== undefined ? json.real : json.error;
    document.getElementById("prediction-result3").innerText =
        json.prediction !== undefined ? json.prediction : json.error;
});
