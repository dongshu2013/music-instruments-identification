$(document).ready(function () {
  $("#back-top").hide();

  $(function () {
    $(window).scroll(function () {
      if ($(window).scrollTop() > $(window).height()/2) {
        $("#back-top").fadeIn(1500);
      }
      else {
        $("#back-top").fadeOut(1000);
      }
    });

    $("#back-top").click(function () {
      $('body,html').animate({ scrollTop: 0 }, 800);
      return false;
    });
  });

  $('.level-1').click(function(e) {
    console.log("On click level-1");
    e.preventDefault();
    var $this = $(this);
    var $collapse = $this.closest('.collapse-group').find('.collapse');
    $collapse.collapse('toggle');
  });

  $('.service-block').hover(function(e) {
    $(this).find('.service-cover').slideToggle();
  }, function(e) {
    $(this).find('.service-cover').slideToggle();
  });

  $('#upload').click(function(e) {
    event.preventDefault();
    var formElement = document.querySelector("form");
    var formData = new FormData(formElement);
    $('#upload').attr('disabled', 'disabled');
    $('#ajax-waiter').show();
    $.ajax({
      url: "/upload",
      method: "POST",
      contentType: false,
      processData: false,
      data: formData,
    }).done(function(data, status) {
      console.log(data)
      $('#ajax-waiter').hide();
      $("#music").css("display","inline");
      $("#analyze").show();
      $('#analyze-result').val('');
      $('#analyze-result').hide();
      $('#upload').attr('disabled', false);

      var audio = $('#music');
      $("#music-source").attr("src", data);
      /****************/
      audio[0].pause();
      audio[0].load();
      /****************/
    });
  });

  $('#analyze-btn').click(function(e) {
    $('#ajax-waiter').show();
    $('#analyze-btn').attr('disabled', 'disabled');
    var src = $("#music-source").attr("src").split("/");
    var name = src[src.length - 1];
    $.ajax({
      url: "/analyze/" + name,
      method: "GET",
    }).done(function(data, status) {
      $('#ajax-waiter').hide();
      $('#analyze-btn').attr('disabled', false);
      $('#analyze-result').val(data);
      $('#analyze-result').show();
    });
  });
});

