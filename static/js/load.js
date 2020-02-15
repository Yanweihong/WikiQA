function query_load() {
    $.ajax({
        url: "/upload",
        type: "post",
        data: {
            query: $("#query_input").val()
        },
        dataType: 'json',
    }).done(function (data) {
        $(".article").each(function (index) {
            $(this).text(data.ids[index])
        })
        $(".score").each(function (index) {
            $(this).text(data.scores[index])
        })
    })
}