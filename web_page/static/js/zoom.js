function set_len_coords(event, imgID, lenID) {
    img = document.getElementById(imgID);
    lens = document.getElementById(lenID);
    imgRect = img.getBoundingClientRect();
    img_left = imgRect.left;
    img_top = imgRect.top;

    /* left coordinate */
    x = event.pageX;
    x = x - imgRect.left;
    x = x - window.scrollX;
    x = x - (lens.offsetWidth / 2);
    if (x > img.width - lens.offsetWidth) { x = img.width - lens.offsetWidth; }
    if (x < 0) { x = 0; }

    /* top coordinate  */
    y = event.pageY;
    y = y - imgRect.top;
    y = y - window.scrollY;
    y = y - (lens.offsetWidth / 2);
    if (y > img.height - lens.offsetHeight) { y = img.height - lens.offsetHeight; }
    if (y < 0) { y = 0; }
    return { "x": x, "y": y }
}

function moveLens(event, elementIDS) {
    const elementID = event.target.id;
    const imgID = elementIDS[elementID]["imgID"];
    const lenID = elementIDS[elementID]["lenID"];
    const coords = set_len_coords(event, imgID, lenID);
    Object.keys(elementIDS).forEach(function(key) {
        var lens = document.getElementById(elementIDS[key]["lenID"]);
        var img = document.getElementById(elementIDS[key]["imgID"]);
        var zoom_port = document.getElementById(elementIDS[key]["portID"]);
        if(lens){
            // Ajustar la posición de la lente
            lens.classList.replace("d-none","d-flex"); 
            lens.style.left = coords.x + "px";
            lens.style.top = coords.y + "px";    
            
            // Ajustar las propiedades de fondo del div de zoom        
            var cx = zoom_port.offsetWidth / lens.offsetWidth;
            var cy = zoom_port.offsetHeight / lens.offsetHeight;
            
            zoom_port.style.backgroundImage = "url('" + img.src + "')";
            zoom_port.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";
            zoom_port.style.backgroundPosition = "-" + (coords.x * cx) + "px -" + (coords.y * cy) + "px";
        }
    });
}

(() => {
    const elementIDS = {
        "pre-zoom-div": {"imgID": "pre-preview", "lenID": "pre-lens", "portID": "pre-zoom-port"},
        "post-zoom-div": {"imgID": "post-preview", "lenID": "post-lens", "portID": "post-zoom-port"},
        "out-zoom-div": {"imgID": "out-img", "lenID": "out-lens", "portID": "out-zoom-port"},
    };

    const addListeners = (key) => {
        const div = document.getElementById(key);
        if (div) {  // Asegurarse de que el elemento existe
            div.addEventListener('mousemove', (event) => moveLens(event, elementIDS));
            div.addEventListener('touchmove', (event) => moveLens(event, elementIDS));
            return true; // Indicar que se han agregado los listeners
        }
        return false; // Indicar que no se encontró el div
    };

    Object.keys(elementIDS).forEach((key) => {
        if (!addListeners(key)) {
            const observer = new MutationObserver(() => { 
                if (addListeners(key)) { observer.disconnect(); }
            });

            // Observa el padre de los elementos que estás intentando agregar
            const parent = document.getElementById('out-img-container'); 
            observer.observe(parent, { childList: true, subtree: true }); 
        }
    });
})();