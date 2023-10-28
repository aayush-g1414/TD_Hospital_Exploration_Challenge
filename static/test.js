fetch('http://127.0.0.1:5555/death_probability', {
    method: 'POST'
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error(error));
