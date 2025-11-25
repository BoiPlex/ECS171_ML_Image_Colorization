const form = document.getElementById('upload-form');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const loadingProgress = document.getElementById('loading-progress');
const fileInput = document.getElementById('image-input');
const uploadBubble = document.getElementById('upload-bubble');
const originalPreview = document.getElementById('original-preview');
const colorizedPreview = document.getElementById('colorized-preview');
let selectedFile = null; // will hold the uploaded file until server returns

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
    selectedFile = files[0];
    console.log('File dropped:', selectedFile.name);
    // Auto-submit form on drop
    form.dispatchEvent(new Event('submit'));
  }
});

// Handle file input change (when user selects from file picker)
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    selectedFile = fileInput.files[0];
    console.log('File selected:', selectedFile.name);
    form.dispatchEvent(new Event('submit'));
  }
});

// Keep track of created object URLs to revoke them
let originalURL = null;
let colorizedURL = null;

// Async version that calls a callback when done (doesn't show result area)
function setPreprocessedPreviewAsync(file, targetW, targetH, callback) {
  if (!file) { callback(); return; }
  if (originalURL && originalURL.startsWith('blob:')) {
    try { URL.revokeObjectURL(originalURL); } catch (e) {}
    originalURL = null;
  }
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, targetW, targetH);
      try {
        const imageData = ctx.getImageData(0, 0, targetW, targetH);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];
          const y = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
          data[i] = data[i + 1] = data[i + 2] = y;
        }
        ctx.putImageData(imageData, 0, 0);
      } catch (e) {
        console.warn('Could not convert to grayscale in canvas:', e);
      }
      const dataUrl = canvas.toDataURL('image/png');
      originalURL = dataUrl;
      originalPreview.src = dataUrl;
      callback(); // signal done, don't show result area here
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

// Handle form submit
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  // Show loading
  loadingDiv.style.display = 'block';
  loadingProgress.style.width = '0%';
  // do not show result area yet — wait until server returns
  // clear only the colorized preview while processing (don't remove DOM nodes)
  if (colorizedPreview) colorizedPreview.src = '';

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

    // Create Image from blob to learn its intrinsic dimensions
    const colorImg = new Image();
    const blobUrl = url; // will be revoked later
    colorImg.onload = () => {
      const w = colorImg.naturalWidth;
      const h = colorImg.naturalHeight;

      // regenerate the original preview resized+grayscaled to match returned image size
      // Use async version with callback to ensure both previews are set before showing result area
      const srcFile = selectedFile || file;
      setPreprocessedPreviewAsync(srcFile, w, h, () => {
        setTimeout(() => { // wait for loading bar animation
          loadingDiv.style.display = 'none';
          // set colorized preview src
          if (colorizedURL) {
            URL.revokeObjectURL(colorizedURL);
            colorizedURL = null;
          }
          colorizedURL = blobUrl;
          colorizedPreview.src = blobUrl;
          // now show the result area with both previews ready
          resultDiv.style.display = 'flex';
          // revoke the URL after image loads to free memory
          colorizedPreview.onload = () => {
            try { URL.revokeObjectURL(blobUrl); } catch (e) {}
          };
        }, 2100); // match total progress duration
      });
    };
    colorImg.src = url;

  } catch (err) {
    console.error(err);
    loadingDiv.style.display = 'none';
    resultDiv.textContent = 'Error processing image.';
  }
});