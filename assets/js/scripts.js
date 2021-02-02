const api_url = "http://127.0.0.1:5000";

$(document).ready(function () {

    let url_word_cloud
    $('#enviar').click(function () {
        if($('input[name=query_param]').val().length == 0){
            alert("Digite algo, por favor!")
            return
        }
        $.ajax({
            url: api_url + "/execute",
            method: "POST",
            data: {
                query_param: $('input[name=query_param]').val()
            },

            beforeSend: function () {
                $('#loading-frame').addClass('d-flex justify-content-center align-items-center');
            },
            
            success: function (response) {
                $('#loading-frame').removeClass('d-flex justify-content-center align-items-center');

                let list_ol = ''
                for (elem of response.data) {
                    classe = elem["SA NLTK"] == -1 ? "class='text-danger'" : "";
                    list_ol += `<li ${classe}>${elem.Tweets}</li>`;
                }
                d = new Date();

                $("#list_tweets").html(list_ol)
                $('#list_tweets li').mark($('input[name=query_param]').val(), {accuracy: "exactly"})
                $("#url_word_cloud").css("background-image", `url('${response.img_wd_path}?${d.getTime()}')`)
                $("#url_graphic").css("background-image", `url('${response.img_gp_path}?${d.getTime()}')`)
                $("#url_line_time").css("background-image", `url('${response.img_tl_path}?${d.getTime()}')`) 
                $("#html_heat_map").html(`<iframe src = ${response.html_hm_path}?${d.getTime()} height = '100%' width ='100%' ></iframe>` )      

            },
            error: function (response) {
                alert('Não foi possivel conectar ao servidor')
                $('#loading-frame').removeClass('d-flex justify-content-center align-items-center');
            }
        });
    })
})
