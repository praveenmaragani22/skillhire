document.addEventListener("DOMContentLoaded", () => {

  const ctx = document.getElementById("skillChart");

  // If resume not uploaded → chart not shown
  if (!ctx) return;

  // Read data passed from backend
  const labels = JSON.parse(ctx.dataset.labels);
  const values = JSON.parse(ctx.dataset.values);

  new Chart(ctx, {
    type: "radar",
    data: {
      labels: labels,
      datasets: [{
        label: "Your Skills",
        data: values,
        borderWidth: 2,
        fill: true
      }]
    },
    options: {
      scales: {
        r: {
          beginAtZero: true,
          max: 100
        }
      }
    }
  });

});
