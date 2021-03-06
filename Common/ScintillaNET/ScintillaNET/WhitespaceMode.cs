#region Using Directives

using System;

#endregion Using Directives


namespace ScintillaNET
{
    /// <summary>
    ///     Specifies the display mode of whitespace characters.
    /// </summary>
    public enum WhitespaceMode
    {
        /// <summary>
        ///     The normal display mode with whitespace displayed as an empty background color.
        /// </summary>
        Invisible = 0,

        /// <summary>
        ///     Whitespace characters are drawn as dots and arrows.
        /// </summary>
        VisibleAlways = 1,

        /// <summary>
        ///     Whitespace used for indentation is displayed normally but after the first visible character, it is shown as dots and arrows.
        /// </summary>
        VisibleAfterIndent = 2,
    }
}
