const form = document.getElementById('upload-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById('image-input');
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/colorize', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Upload failed');

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    resultDiv.innerHTML = `<h2>Colorized Image:</h2><img src="${url}" alt="Colorized">`;
  } catch (err) {
    console.error(err);
    resultDiv.textContent = 'Error processing image.';
  }
});
