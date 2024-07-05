const all_language = {
  "es":{
      "title":"Clasificación de daños en imágenes satelitales",
      "lang-switch":"Cambiar a Inglés",
      "project":"Proyecto final de carrera de Licenciatura en Ciencias de la Computación",
      "pre":"Imagen previa al desastre natural",
      "post":"Imagen posterior al desastre natural",
      "process": "Procesar imágenes",
      "output": "Imagen de salida",
      "settings" : "Superposición",
      "table": "Conteo de edificios por daños",
      "level": "Nivel",
      "class" : "Tipo",
      "class-1": "Sin daños",
      "class-2": "Daño menor",
      "class-3": "Daño mayor",
      "class-4": "Destruido",
      "class-5": "Sin clasificar",
      "saturation": "saturación",
      "apply": "aplicar"
  },
  "en":{
      "title":"Damage assessment in satellite images",
      "lang-switch":"Change to Spanish",
      "project":"Undergraduate Final Project for a Bachelor's Degree in Computer Science",
      "pre":"Pre-disaster image",
      "post":"Post-disaster image",
      "process": "Process images",
      "output": "Process output",
      "settings" : "Superposition",
      "table": "Building count by damage",
      "level": "Level",
      "class" : "Type",
      "class-1": "No damage",
      "class-2": "Minor damage",
      "class-3": "Mayor damage",
      "class-4": "Destroyed",
      "class-5": "Not classified",
      "saturation": "saturation",
      "apply": "apply"
  }
}
const language = all_language[window.lang]
  
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
              ${language[level.id]} ${language['saturation']}
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
      cell1.textContent = language[level['id']];
      cell2.textContent = level['num'];
  }
}

function set_images(mask_obj, bbs_arr) {
  const out_img = document.getElementById('output-preview');
  out_img.src = "";
  out_img.src = mask_obj["filename"];

  const container = document.getElementById('out-img-container');

  // Remover imágenes existentes
  const existingImages = container.querySelectorAll('img[id^="bb-"]');
  existingImages.forEach(img => img.remove());

  // Agregar nuevas imágenes dinámicamente
  for (let i = 0; i < bbs_arr.length; i++) {
      const bb_img = document.createElement('img');
      bb_img.id = `bb-${i}`;
      bb_img.src = bbs_arr[i]["filename"];
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
    .catch(error => {console.error('Error:', error)});
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

