$(document).ready(function(){
firebase.auth().onAuthStateChanged(function(user)) {
  if (user) {
  //User is signed in
  var token = firebase.auth().currentUser.uid;
  queryDatabase(token);
  } else {
  //No user is signed in
    window.location = "index.html";
  }
});
});

function queryDatabase(token) {
firebase.database().ref('/returns/').once('value').then(function(snapshot)) {
var returnsObject = snapshot.val();
console.log(returnsArray);
var keys = Object.keys(returnsObject);
});

}
