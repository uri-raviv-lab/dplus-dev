param($version)

# $version = refs/tags/dplus-v4.6.1.0

$version_split = $version.Split("/")
$version_str = $version_split[2]

$version_num = $version_str.replace("dplus-v", "")
$version_num_split = $version_num.Split(".")

$major = $version_num_split[0]
$minor = $version_num_split[1]
$revision = $version_num_split[2]
$build = $version_num_split[3]

$file_content = Get-Content .\frontend_version.h
$i = 0

$new_file_content = $file_content
foreach($line in $file_content) 
{
    if ($line.Contains("#define FRONTEND_VERSION_MAJOR"))
    {
        $line_no_comment = $line.Split("//")[0]
        $curr = $line_no_comment.Split(" ")[2]
        $new_line = $line.Replace($curr, [string]$major)
        $new_file_content = $new_file_content.Replace($line, $new_line)
    }

    if ($line.Contains("#define FRONTEND_VERSION_MINOR"))
    {
        $line_no_comment = $line.Split("//")[0]
        $curr = $line_no_comment.Split(" ")[2]
        $new_line = $line.Replace($curr, [string]$minor)
        $new_file_content = $new_file_content.Replace($line, $new_line)
    }

    if ($line.Contains("#define FRONTEND_VERSION_REVISION"))
    {
        $line_no_comment = $line.Split("//")[0]
        $curr = $line_no_comment.Split(" ")[2]
        $new_line = $line.Replace($curr, [string]$revision)
        $new_file_content = $new_file_content.Replace($line, $new_line)
    }

    if ($line.Contains("#define FRONTEND_VERSION_BUILD"))
    {
        $line_no_comment = $line.Split("//")[0]
        $curr = $line_no_comment.Split(" ")[2]
        $new_line = $line.Replace($curr, [string]$build)
        $new_file_content = $new_file_content.Replace($line, $new_line)
    }
}

Set-Content -Path .\frontend_version.h -Value $new_file_content

