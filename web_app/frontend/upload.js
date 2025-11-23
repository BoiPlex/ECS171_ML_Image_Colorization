const form = document.getElementById('upload-form');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const loadingProgress = document.getElementById('loading-progress');
const fileInput = document.getElementById('image-input');
const uploadBubble = document.getElementById('upload-bubble');

// Custom cursor
const cursor = document.createElement('div');
cursor.style.width = '20px';
cursor.style.height = '20px';
cursor.style.borderRadius = '50%';
cursor.style.background = '#555';
cursor.style.position = 'fixed';
cursor.style.pointerEvents = 'none';
cursor.style.zIndex = 9999;
cursor.style.transition = 'transform 0.05s ease-out';
cursor.style.transform = 'translate(-50%, -50%)';
document.body.appendChild(cursor);

document.addEventListener('mousemove', e => {
  cursor.style.left = e.clientX + 'px';
  cursor.style.top = e.clientY + 'px';
});

// Bubble hover effect on upload
uploadBubble.addEventListener('mouseenter', () => {
  cursor.style.transform = 'translate(-50%, -50%) scale(1.5)';
});
uploadBubble.addEventListener('mouseleave', () => {
  cursor.style.transform = 'translate(-50%, -50%) scale(1)';
});

// Click to open file picker
uploadBubble.addEventListener('click', () => fileInput.click());

// Drag & drop handling
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
  if (files.length > 0) {
    fileInput.files = files;
    console.log('File dropped:', files[0].name);
    // Auto-submit form on drop
    form.dispatchEvent(new Event('submit'));
  }
});

// Handle file input change (when user selects from file picker)
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    console.log('File selected:', fileInput.files[0].name);
    form.dispatchEvent(new Event('submit'));
  }
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