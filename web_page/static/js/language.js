function setLanguage(lang) {
    document.documentElement.lang = lang; // Cambia el atributo lang del elemento <html>
    document.getElementById("title").innerText = translations[lang].title;
    document.getElementById("greeting").innerText = translations[lang].greeting;
  }