// -------------------------------
// Handle category clicks (filter logic)
const categories = document.querySelectorAll('.categories li');
const projectCard = document.querySelector('.project-card');

categories.forEach(cat => {
  cat.addEventListener('click', () => {
    // Remove .active from all categories
    categories.forEach(c => c.classList.remove('active'));
    cat.classList.add('active');

    const selected = cat.textContent.trim();
    // Currently only one project; adjust if you have multiple tags
    if (selected === 'All' || selected === 'Design') {
      projectCard.style.display = 'block';
    } else {
      projectCard.style.display = 'none';
    }
  });
});

// -------------------------------
// Create custom cursor bubble
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

// Update cursor position
document.addEventListener('mousemove', e => {
  cursor.style.left = e.clientX + 'px';
  cursor.style.top = e.clientY + 'px';
});

// -------------------------------
// Enlarge cursor on hover for project cards
document.querySelectorAll('.project-card').forEach(card => {
  card.addEventListener('mouseenter', () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1.5)';
  });
  card.addEventListener('mouseleave', () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1)';
  });
});

// -------------------------------
// Enlarge cursor on hover for category menu links
document.querySelectorAll('.categories li a').forEach(link => {
  link.addEventListener('mouseenter', () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1.3)';
  });
  link.addEventListener('mouseleave', () => {
    cursor.style.transform = 'translate(-50%, -50%) scale(1)';
  });
});

// -------------------------------
// Highlight current page menu link dynamically
const menuLinks = document.querySelectorAll('.categories li a');
menuLinks.forEach(link => {
  // Compare the link href to the current page URL
  if (link.href === window.location.href) {
    link.classList.add('active');
  } else {
    link.classList.remove('active');
  }
});
