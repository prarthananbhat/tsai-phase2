function uploadAndclassifyImage(){
    var fileInput = document.getElementById('resnet34FileUpload').files;
    if(!fileInput.length){
        return alert("Please choose a file to upload first");
    }
    var file = fileInput[0]
    var filename = file.name

    var formData = new FormData();
    formData.append(filename,file)

    console.log(filename)

    $.ajax({
        async : true,
        crossDomain : true,
        method : 'POST',
        url : 'https://ic87evu6q3.execute-api.ap-south-1.amazonaws.com/dev/classify_image',
        data : formData,
        processData : false,
        contentType : false,
        mimeType : "multipart/form-data"
    }).done(function (response) {
        console.log(response);
        document.getElementById('result').textContent = response;
    }).fail(function () {alert ("There was an error while sending a prediction request");
    });
};
$('#btnResNetUpload').click(uploadAndclassifyImage())