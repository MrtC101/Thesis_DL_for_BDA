// PROCESS FUNCTIONS
function build_settings(table_arr){
  table = document.getElementById("setting-t")
  while (table.rows.length > 1) {
    table.deleteRow(1);
  }
  for (const level of table_arr) {
      const row = table.insertRow();
      const cell1 = row.insertCell(0);
      console.log()
      cell1.innerHTML = `
      <div class="row block">
          <label class="col-6" for="${level.id}">
              ${window.all_language[window.lang][level.id]}
          </label>
          <input class="col-4" type="range" id="${level.id}" min="0" max="1" step="0.1" value="1">
          <output class="col-2" id="${level.id}-val" for="${level.id}">1</output>
      </div>
      `;
      }
}

function build_table(table_arr) {
  const table = document.getElementById('count-tbl');
  // Eliminar todas las filas de la tabla
  while (table.rows.length > 1) {
      table.deleteRow(1);
  }
  for (const level of table_arr) {
      const row = table.insertRow();
      const cell1 = row.insertCell(0);
      const cell2 = row.insertCell(1);
      cell1.style.backgroundColor = level['color'];
      cell1.textContent = window.all_language[window.lang][level['id']];
      cell2.textContent = level['num'];
  }
}

function set_images(mask_obj, bbs_arr) {
  const out_img = document.getElementById('output-preview');
  out_img.src = mask_obj;

  const container = document.getElementById('out-img-container');

  // Remover imágenes existentes
  const existingImages = container.querySelectorAll('img[id^="bb-"]');
  existingImages.forEach(img => img.remove());

  for(const path of bbs_arr){
    const bb_img = document.createElement('img');
    i = path.split("_")[1];
    bb_img.id = `bb-${i}`;
    bb_img.src = path;
    bb_img.style.zIndex = i;
    bb_img.style.filter = "opacity(1)";
    container.appendChild(bb_img);
  }
}

/**
 * 
 * @returns {object}
 */
async function get_images(){
  preElement = document.getElementById("pre-preview")
  const pre_img = await fetch(preElement.src)
    .then(response => response.blob());
  
  postElement = document.getElementById("post-preview")
  const post_img = await fetch(postElement.src)
    .then(response => response.blob());

  const formData = new FormData();
  formData.append('pre-img', pre_img, 'pre_img.png');
  formData.append('post-img', post_img, 'post_img.png');
  const options = {
    method: 'POST',
    body: formData
  }
  return options
}


//code for image superposition
/**
 * @function
 * @description This listener changes the opacity attribute for each image.
 * @this HTMLElement
 */
function changeOpListener() {
  document.querySelectorAll('input[type="range"]').forEach(
    (range) => {
      n = range.getAttribute("id")
      i = n.charAt(n.length-1)
      bb = document.getElementById(`bb-${i}`);
      bb.style.filter = `opacity(${range.value})`;
    }
  );
}

/**
 * @description This function add listeners for each setting.
 */
function addSettingsListeners(){
  // Actualiza el valor del output cuando cambia el input range
  document.querySelectorAll('input[type="range"]').forEach(
    range => {
      const output = document.querySelector(`#${range.id}-val`);
      range.addEventListener('input', () =>{output.textContent = range.value;});
      range.addEventListener('input', changeOpListener)
    }
  );
}


/**
 * @description This method calls the server for a prediction.
 */
async function process(){
  //Loading bar
  const overlay = document.getElementById("overlay");
  overlay.style.display = "flex";
  const serverUrl = window.predictURL
  options = await get_images()
  fetch(serverUrl, options)
    .then(async response =>{
      dict = await response.json()
      build_table(dict["table"])
      build_settings(dict["table"])
      set_images(dict["mask"], dict["bbs"])
      addSettingsListeners()
    })
    .catch(error => {console.error('Error:', error)})
    .finally(() => overlay.style.display = "none")
}


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

/**
 * @description This function changes the language of the page
 */
function switchLang() {
  const serverUrl = window.location.href
  window.lang = window.lang == "es" ? "en" : "es"
  const options = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({"lang": window.lang})
  }

  function initscripts(){
    const scripts = document.querySelectorAll('script');
    scripts.forEach(script => {
        const newScript = document.createElement('script');
        newScript.src = script.src;
        document.body.appendChild(newScript);
        script.remove();
    });
  }

  fetch(serverUrl, options)
    .then(async response =>{
      // Get the root element of the HTML document
      const rootElement = document.documentElement;
      rootElement.innerHTML = await response.text();
      initscripts()
    })
    .catch(error => {console.error('Error:', error)});
}

