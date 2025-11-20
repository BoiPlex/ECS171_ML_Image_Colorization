const categories = document.querySelectorAll('.categories li');
const projectCard = document.querySelector('.project-card');

categories.forEach(cat => {
  cat.addEventListener('click', () => {
    // remove .active from all
    categories.forEach(c => c.classList.remove('active'));
    cat.classList.add('active');

    const selected = cat.textContent.trim();
    // For now, we only have one project.
    // You could check project tags, but just show/hide:
    if (selected === 'All' || selected === 'Design' /* or other tag logic */ ) {
      projectCard.style.display = 'block';
    } else {
      projectCard.style.display = 'none';
    }
  });
});
