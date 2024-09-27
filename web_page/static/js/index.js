function changeLanguage(lang) {
  curr_lang = document.documentElement.lang
  next_lang = curr_lang == "en" ? "es" : "en"
  window.location.replace(`/${next_lang}`);
}

// Change theme listener
(() => {
  const themeToggleButton = document.getElementById('theme-switcher');
  themeToggleButton.addEventListener('click', () => {
    const currentTheme = getStoredTheme();
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark'; // Alternar entre temas
    setStoredTheme(newTheme);
    setTheme(newTheme);
  });
})();
