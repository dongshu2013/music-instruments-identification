var express = require('express');
var app = express();
var port = process.env.PORT || 9000;
//var app = require('./servers/main-server')
var app = require('./server')
app.listen(port);

module.exports = app;
