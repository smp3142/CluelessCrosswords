if ( -not (Test-Path 'CluelessCrosswords.py' -PathType Leaf) ) { "Please run from the folder that contains the CluelessCrosswords.py file." }
else {
    "__pycache__", "build", "dist" | ForEach-Object -Process {if (Test-Path $_) { Remove-Item -r $_ } }
    if (Test-Path CluelessCrosswords.spec) { Remove-Item CluelessCrosswords.spec }
    pyinstaller --add-data "words.pkl.gz;." --add-data "useful.pkl.gz;." --noconsole CluelessCrosswords.py
    Compress-Archive -Force -Path ./dist/CluelessCrosswords -DestinationPath ../binaries/windows.zip
    Remove-Item CluelessCrosswords.spec
    "__pycache__", "build", "dist" | ForEach-Object -Process {if (Test-Path $_) { Remove-Item -r $_ } }
}
