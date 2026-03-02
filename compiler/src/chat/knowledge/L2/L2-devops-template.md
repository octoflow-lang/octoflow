# template (L2)
devops/template — Simple {key} placeholder template rendering.

## Functions
render(template: string, vars: map) → string
  Replace {key} placeholders with values
load_template(path: string) → string
  Load template string from file
render_file(path: string, vars: map) → string
  Load and render template from file
