Dropzone.autoDiscover = false;

function init() {
    //create dropezone opject 
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drag your image here",
        autoProcessQueue: false
    });
    //when file is added execute this fucntion 
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });
    //when the upload is complete 
    dz.on("complete", function (file) {
        //get base64 string of image data
        let imageData = file.dataURL;
        //URL where my flask server is running 
        var url = "http://127.0.0.1:5000/classify_image";
        //post http request post(url,input data, response)
        $.post(url, {
            image_data: file.dataURL
        },function(data, status) {
            console.log(data);
            if (!data || data.length==0) {
                $("#result").hide();
                $("#divClassTable").hide();                
                $("#error").show();
                return;
            }
            let players = ["lionel_messi", "malek_jaziri", "maria_sharapova", "ons_jabeur", "rafael_nadal"];
            let match = null;
            let bestScore = -1;
            for (let i=0;i<data.length;++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if(maxScoreForThisClass>bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }
            if (match) {
                $("#error").hide();
                $("#result").show();
                $("#divClassTable").show();
                $("#result").html($(`[data-player="${match.class}"`).html());
                let classDictionary = match.class_dictionary;
                for(let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let proabilityScore = match.class_probability[index];
                    let elementName = "#score_" + personName;
                    $(elementName).html(proabilityScore);
                }
            }
            // dz.removeFile(file);            
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}



$(document).ready(function() {
    console.log( "ready!" );
    $("#error").hide();
    $("#result").hide();
    $("#divClassTable").hide();

    init();
});