const form = document.getElementById('upload-form');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const loadingProgress = document.getElementById('loading-progress');
const fileInput = document.getElementById('image-input');
const uploadBubble = document.getElementById('upload-bubble');

// Click or drag behavior already handled
uploadBubble.addEventListener('click', () => fileInput.click());
uploadBubble.addEventListener('dragover', e => {
  e.preventDefault();
  uploadBubble.classList.add('dragover');
});
uploadBubble.addEventListener('dragleave', e => {
  e.preventDefault();
  uploadBubble.classList.remove('dragover');
});
uploadBubble.addEventListener('drop', e => {
  e.preventDefault();
  uploadBubble.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) fileInput.files = files;
});

// Handle form submit
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  // Show loading
  loadingDiv.style.display = 'block';
  resultDiv.innerHTML = '';
  loadingProgress.style.width = '0%';

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/colorize', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) throw new Error('Upload failed');

    // Simulate loading bar for demo purposes
    let width = 0;
    const interval = setInterval(() => {
      if (width >= 100) clearInterval(interval);
      width += 5; 
      loadingProgress.style.width = width + '%';
    }, 100);

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);

    setTimeout(() => { // wait for loading bar animation
      loadingDiv.style.display = 'none';
      resultDiv.innerHTML = `<h2>Colorized Image:</h2><img src="${url}" alt="Colorized" class="result-image">`;
    }, 2100); // match total progress duration

  } catch (err) {
    console.error(err);
    loadingDiv.style.display = 'none';
    resultDiv.textContent = 'Error processing image.';
  }
});