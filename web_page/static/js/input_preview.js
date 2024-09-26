/**
 * @function
 * @name previewListener
 * @description This function is a listener that changes the image preview.
 * @this {HTMLElement}
 * @param {Event} ev
 * @returns {*}
 */
function previewListener(ev) {
    const inputElement = ev.target;
    const file = inputElement.files[0];
    if (file) {
      const fileReader = new FileReader();
      const id = this.getAttribute("for");
      const preview = document.getElementById(id);
      fileReader.onload = event => {
        preview.setAttribute("src", event.target.result);
      };
      fileReader.readAsDataURL(file);
    }
  }

function addUploadListeners(){
// Code for loading images
const pre_in = document.getElementById("pre-input");
const post_in = document.getElementById("post-input");
pre_in.addEventListener("change", previewListener);
post_in.addEventListener("change", previewListener);
}
    
addUploadListeners()