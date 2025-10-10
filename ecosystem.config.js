module.exports = {
  apps: [
    {
      name: "face-search-api",
      script: "app.py",  // your Flask entrypoint
      interpreter: "python",
      cwd: "./",
      watch: false,
      env: {
        FLASK_ENV: "production",
        FLASK_DEBUG: "False",
      },
    },
  ],
};

