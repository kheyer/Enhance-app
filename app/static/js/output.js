var el = x => document.getElementById(x);

var img_enc = document.getElementById('img_b64');

var img_blob = dataURLtoBlob(img_enc.value); // create blob
var image_box = document.getElementById('myImage');
var url = URL.createObjectURL(img_blob);
image_box.src = url

$('body').on('click', '#myImage', function(event) {
    var link = document.createElement('a');
    var url = URL.createObjectURL(img_blob);
    // Add the element to the DOM
    link.setAttribute("type", "hidden");
    link.href = url;
    link.target = "_blank"; // set to open in new tab
    document.body.appendChild(link); // append element
    link.click(); // activate
    link.remove(); // remove
});


$('body').on('click', '#submit_btn', function(event) {
    var link = document.createElement('a');
    var url = URL.createObjectURL(img_blob);
    // Add the element to the DOM
    link.setAttribute("type", "hidden");
    link.download = 'enhanced_image.png'; // set to download
    link.href = url;
    document.body.appendChild(link); // append element
    link.click(); // activate
    link.remove(); // remove
});



function dataURLtoBlob(dataURL) {
    // Decode the dataURL
    var imageType = dataURL.split(',')[0];     
    var binary = atob(dataURL.split(',')[1]);
    // Create 8-bit unsigned array
    var array = [];
    for(var i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
    // Return our Blob object
    // console.log("imageType", imageType);

    if (imageType.indexOf("jpeg") >=0) {
        return new Blob([new Uint8Array(array)], {type: 'image/jpeg'});
    }
    else {
        return new Blob([new Uint8Array(array)], {type: 'image/png'});
    }
  }