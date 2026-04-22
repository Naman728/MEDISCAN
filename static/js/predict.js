(function () {
  const form = document.querySelector('.predict-form');
  const resultBox = document.querySelector('.result-box');

  if (!form || !resultBox) {
    return;
  }

  form.addEventListener('submit', async function (event) {
    event.preventDefault();

    const fileInput = form.querySelector('input[name="file"]');
    if (!fileInput || !fileInput.files || !fileInput.files.length) {
      resultBox.textContent = 'Please choose an image file first.';
      return;
    }

    const endpoint = form.dataset.endpoint;
    const payload = new FormData();
    payload.append('file', fileInput.files[0]);

    resultBox.textContent = 'Running prediction...';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: payload
      });

      const data = await response.json();
      if (!response.ok) {
        resultBox.textContent = `Error (${response.status}): ${JSON.stringify(data, null, 2)}`;
        return;
      }

      resultBox.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
      resultBox.textContent = `Request failed: ${error.message}`;
    }
  });
})();