var fs = require('fs');
var formidable = require('formidable');
var url = require('url');
var exec = require('child_process').exec, child;

exports.uploadFile = function(req, res, callback) {
  var form = new formidable.IncomingForm();
  form.parse(req, function(err, fields, files) {
    if (err) {
      console.error(err.message);
      return;
    }

    fs.readFile(files.audio.path, function (err, data) {
      var audioName = files.audio.name
      if(!audioName){
        console.log("There was an error")
        res.redirect("/");
        res.end();
      } else {
        var newPath = __dirname + "/../public/music/" + audioName;
        fs.writeFile(newPath, data, function (err) {
          res.writeHead(200, {'content-type': 'text/plain'});
          res.end('/music/' + audioName);
        });
      }
    });
  });
}

exports.streamAudio = function(name, req, res, callback) {
  var filePath = __dirname + "/../public/music/" + name;
  var stat = fs.statSync(filePath);

  res.writeHead(200, {
    'Content-Type': 'audio/wav',
    'Content-Length': stat.size
  });

  var readStream = fs.createReadStream(filePath);
  readStream.pipe(res);
}

exports.analyze = function(name, req, res, callback) {
  var filePath = __dirname + "/../public/music/" + name;
  var codePath = __dirname + "/../../code/IdentifyingInstrument.py";
  var modelPath = __dirname + "/../../code/models/rbfSVM_dim390_model.pkl";
  var stat = fs.statSync(filePath);

  var child = exec('python '+ codePath + ' ' + filePath + ' ' + modelPath,
    function (error, stdout, stderr) {
        console.log('stdout: ' + stdout);
        result = stdout;
        res.writeHead(200, {'content-type': 'text/plain'});
        res.end(result);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });
}
