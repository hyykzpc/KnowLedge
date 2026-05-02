<%*
const file = app.workspace.getActiveFile();
let content = await app.vault.read(file);

content = content.replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, function(match, formula) {
  return "$" + "$\n" + formula.trim() + "\n" + "$" + "$";
});

content = content.replace(/\\\(\s*([\s\S]*?)\s*\\\)/g, function(match, formula) {
  return "$" + formula.trim() + "$";
});

await app.vault.modify(file, content);
%>