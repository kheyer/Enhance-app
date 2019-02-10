from io import BytesIO
from fastai import *
from fastai.vision import *

def result_html(img_io):
    return f''' <html>
                <head>
                    <title>Enhanced Image</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">

                <!--===============================================================================================-->
                    <link rel="icon" type="image/png" href="static/images/icons/favicon.ico"/>
                    <link rel="stylesheet" type="text/css" href="static/css/main.css">
                <!--===============================================================================================-->
                </head>
                <body>
                    <div class="container-contact100">
                        <div class="wrap-contact100">
                            <a href="/"><img height="34" width="34" src="static/images/25694.png"></a>
                            <span class="contact100-form-title">Resolution Enhanced Image</span>
                            <input id="img_b64" type="hidden" name="img" value="data:image/png;base64,{img_io}">
                            <div>
                                <img id="myImage" class="image-display" alt="enhanced_image">
                            </div>

                            <div class="container-contact100-form-btn">
                                <div class="wrap-contact100-form-btn">
                                    <div class="contact100-form-bgbtn"></div>
                                    <button type = "button" id = "submit_btn" class="contact100-form-btn">

                                        <a id="myButton" href="#" class="contact100-form-btn" target="_blank">
                                            Download
                                        </a>
                                    </button>
                                </div>
                            </div>

                            <div
                                class="navbar">
                                    <a href="https://github.com/kheyer/Enhance-app" target="_blank"><img src="static/images/github.png" target="_blank", height=42, width=42 /></a>
                            </div>
                            <div class="wrap-input100 validate-input">
                            <script src="static/js/output.js"></script>
            '''