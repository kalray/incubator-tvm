#!/usr/bin/env sh

__ScriptVersion="1.0"

# Constants
readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly ROOT_FOLDER="$( readlink -f $SCRIPT_DIR/.. )"

#===  FUNCTION  ================================================================
#         NAME:  usage
#  DESCRIPTION:  Display usage information.
#===============================================================================
usage ()
{
	echo "Usage : $0 <file.dot> [options] [--]

    Options:
    -h|help       Display this message
    -v|version    Display script version"

}    # ----------  end of function usage  ----------

#-----------------------------------------------------------------------
#  Handle command line arguments
#-----------------------------------------------------------------------

while getopts ":hv" opt
do
  case $opt in

	h|help     )  usage; exit 0   ;;

	v|version  )  echo "$0 -- Version $__ScriptVersion"; exit 0   ;;

	* )  echo -e "\n  Option does not exist : $OPTARG\n"
		  usage; exit 1   ;;

  esac    # --- end of case ---
done
shift $(($OPTIND-1))

# Arguments
json_arg="$1"

# Check if json_arg was supplied
if [ -z "$json_arg" ]; then
  usage
  exit -1
fi

# Check if json_arg file really exists
if [ ! -f "$json_arg" ]; then
  echo "Supplied file $json_arg does not exist"
  exit -1
fi

OUT_DOT="$(mktemp -t XXXXXXXX.dot)"
# Invoke OCAML script to generate dot
echo "\
--------------------------------------------------------------------------------
Generating dot file from json file:
$json_arg
to
$OUT_DOT
--------------------------------------------------------------------------------
"

./json2dot.ml "$json_arg" > "$OUT_DOT"

# Invoke sh script that calls dottoxml python script
OUT_ROOT="${OUT_DOT#/tmp/}"
OUT_ROOT="${OUT_ROOT%.dot}"
echo "\
--------------------------------------------------------------------------------
Generating graphml file from dot file in current directory:
$OUT_ROOT.graphml
--------------------------------------------------------------------------------
"

./dot2graph.sh "$OUT_DOT"
