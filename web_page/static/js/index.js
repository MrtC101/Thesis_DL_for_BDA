function changeLanguage(lang) {
  curr_lang = document.documentElement.lang
  next_lang = curr_lang == "en" ? "es" : "en"
  window.location.replace(`/${next_lang}`);
}