const express = require('express');
const path = require('path');

const app = express();

const port = Number(process.env.PORT) || 8080;
const publicDir = path.join(__dirname, 'public');

// Serve static assets (and allow /foo -> /foo.html when present)
app.use(
  express.static(publicDir, {
    extensions: ['html'],
    fallthrough: true
  })
);

app.get('/healthz', (req, res) => {
  res.type('text/plain').send('ok');
});

// Fallback to index.html for any unknown route
app.get('*', (req, res) => {
  res.sendFile(path.join(publicDir, 'index.html'));
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Server listening on http://0.0.0.0:${port}`);
});
