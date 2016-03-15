var express = require('express');
var router = express.Router();
var index = require('./lib/index');

router.use(function(req, res, next) {
  console.log('%s %s %s', req.method, req.url, req.path);
  next();
});

router.get('/', function(req, res, next){
  res.render('index', {
    title: "Music Instrument Indentification"
  });
});

router.post('/upload', function(req, res, next){
  index.uploadFile(req, res);
});

router.get('/music/:name', function(req, res, next){
  var name = req.params.name;
  index.streamAudio(name, req, res);
});

router.get('/analyze/:name', function(req, res, next){
  var name = req.params.name;
  index.analyze(name, req, res);
});

module.exports = router
