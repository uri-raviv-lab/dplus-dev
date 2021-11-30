#define SVN_REV $WCREV$
#define SVN_REV_STR "$WCMODS?$WCREV$M:$WCREV$$"

/*
$WCREV$         
Highest committed revision number

$WCDATE$        
Date of highest committed revision

$WCDATE=$      
Like $WCDATE$ with an added strftime format after the =

$WCRANGE$      
Update revision range

$WCURL$        
Repository URL of the working copy

$WCNOW$        
Current system date & time

$WCNOW=$       
Like $WCNOW$ with an added strftime format after the =

$WCLOCKDATE$   
Lock date for this item

$WCLOCKDATE=$  
Like $WCLOCKDATE$ with an added strftime format after the =

$WCLOCKOWNER$   
Lock owner for this item

$WCLOCKCOMMENT$ 
Lock comment for this item
 
 *
$WCMODS?True:False$       
True if local modifications found

$WCMIXED?True:False$      
True if mixed update revisions found

$WCINSVN?True:False$      
True if the item is versioned

$WCNEEDSLOCK?True:False$   
True if the svn:needs-lock property is set

$WCISLOCKED?True:False$   
True if the item is locked

 */
