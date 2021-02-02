const api_url = "http://127.0.0.1:5000";

function hiliter(word, element) {
    var rgxp = new RegExp(word, 'g');
    var repl = '<span class="highlight">' + word + '</span>';
    element.innerHTML = element.innerHTML.replace(rgxp, repl);
}


$(document).ready(function () {

    $('#positive').html('0%')
    $('#negative').html('0%')

  

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
                $('#positive').html(`${response.positive}%`)
                $('#negative').html(`${response.negative}%`)

                let list_ol = ''
                for (elem of response.data) {
                    classe = elem["SA NLTK"] == -1 ? "class='text-danger'" : "";
                    list_ol += `<li ${classe}>${elem.Tweets}</li>`;
                }
                $("#list_tweets").html(list_ol)
                $('#list_tweets li').mark($('input[name=query_param]').val(), {accuracy: "exactly"})
                d = new Date();
                $("#url_word_cloud").css("background-image", `url('${response.img_path}?${d.getTime()}')`)
             
                
            },
            error: function (response) {
                alert('Não foi possivel conectar ao servidor')
                $('#loading-frame').removeClass('d-flex justify-content-center align-items-center');
            }
        });
    })
})
